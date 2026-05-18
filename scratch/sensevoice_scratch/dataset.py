"""
CTC-only dataset for scratch training — target 앞에 task prefix를 붙이지 않는다.

원본 SenseVoiceCTCDataset은 [lid, emo, event, itn, text] 형식으로 4개의
task token을 target 앞에 prepend한다. scratch (task embedding 없음) 모델에선
이 prefix가 불필요/무해하지 않으므로 제거한다.
"""
import logging
import traceback

import torch

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "SenseVoiceCTCDatasetNoTask")
class SenseVoiceCTCDatasetNoTask(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            cls = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = cls(**kwargs.get("preprocessor_speech_conf"))
        self.preprocessor_speech = preprocessor_speech

        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            cls = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = cls(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.retry = kwargs.get("retry", 5)
        self.max_target_length = kwargs.get("max_target_length", 200)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend
        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

    def get_source_len(self, index):
        return self.index_ds.get_source_len(self.index_ds[index])

    def get_target_len(self, index):
        return self.index_ds.get_target_len(self.index_ds[index])

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        output = None
        for idx in range(self.retry):
            index_cur = index if idx == 0 else torch.randint(0, len(self.index_ds), ()).item()
            item = self.index_ds[index_cur]
            source = item["source"]
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {e}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)
            speech, speech_lengths = extract_fbank(
                data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
            )

            if speech_lengths > self.batch_size:
                continue
            if self.permute:
                speech = speech.permute(0, 2, 1)

            asr_target = item["target"]
            if self.preprocessor_text:
                asr_target = self.preprocessor_text(asr_target)

            target_ids = self.tokenizer.encode(asr_target)
            if len(target_ids) == 0 or len(target_ids) > self.max_target_length:
                continue

            text = torch.tensor(target_ids, dtype=torch.int64)
            text_lengths = torch.tensor([len(target_ids)], dtype=torch.int32)

            output = {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
            }
            break

        return output

    def collator(self, samples):
        outputs = {}
        for sample in samples:
            if sample is None:
                continue
            for key in sample:
                outputs.setdefault(key, []).append(sample[key])

        if len(outputs) < 1:
            logging.error("ERROR: data is empty!")
            return {
                "speech": torch.rand((10, 560), dtype=torch.float32)[None, :, :],
                "speech_lengths": torch.tensor([10], dtype=torch.int32)[:, None],
                "text": torch.tensor([1], dtype=torch.int32)[None, :],
                "text_lengths": torch.tensor([1], dtype=torch.int32)[:, None],
            }

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                pad_value = (
                    self.int_pad_value
                    if data_list[0].dtype in (torch.int64, torch.int32)
                    else self.float_pad_value
                )
                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

        if self.batch_type != "example":
            for i in range(10):
                outputs = self._filter_badcase(outputs, i=i)

        return outputs

    def _filter_badcase(self, outputs, i=0):
        b, t, _ = outputs["speech"].shape
        if b * t > self.batch_size * 1.25:
            beg = torch.randint(0, 2, ()).item()
            if b < 2:
                beg = 0
            logging.info(
                f"Warning, b*t={b*t} > {self.batch_size}, drop half data {i}th, beg={beg}"
            )
            for key in outputs:
                outputs[key] = outputs[key][beg : beg + b : 2]

            sl_max = outputs["speech_lengths"].max().item()
            outputs["speech"] = outputs["speech"][:, :sl_max, :]
            tl_max = outputs["text_lengths"].max().item()
            outputs["text"] = outputs["text"][:, :tl_max]

        return outputs
