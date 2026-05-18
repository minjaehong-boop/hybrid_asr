"""
SenseVoiceSmall CTC-only scratch variant.

원본 SenseVoiceSmall에서 task embedding(LID/SER/AEC/ITN), rich CE loss,
task query concat을 모두 제거한 순수 CTC 모델. encoder 구조는 그대로 유지.
"""
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter


@tables.register("model_classes", "SenseVoiceSmallCTCScratch")
class SenseVoiceSmallCTCScratch(nn.Module):
    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 560,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        if specaug is not None:
            specaug = tables.specaug_classes.get(specaug)(**specaug_conf)
        if normalize is not None:
            normalize = tables.normalize_classes.get(normalize)(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        self.ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        stats = {"loss_ctc": loss_ctc.detach(), "loss": loss_ctc.detach()}

        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss_ctc, stats, batch_size), loss_ctc.device)
        return loss, stats, weight

    def encode(self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs):
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            t1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            t2 = time.perf_counter()
            meta_data["load_data"] = f"{t2 - t1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            t3 = time.perf_counter()
            meta_data["extract_feat"] = f"{t3 - t2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_logits = self.ctc.log_softmax(encoder_out)

        results = []
        b = encoder_out.size(0)
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer["1best_recog"]

        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)
            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()
            text = tokenizer.decode(token_int) if tokenizer is not None else ""
            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text
            results.append({"key": key[i], "text": text})

        return results, meta_data
