from typing import Dict, Any, List, Tuple, Optional
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from simple_xmm.modality_processors.base import BaseModalProcessor
from torch.nn.utils.rnn import pad_sequence


class AudioModalProcessor(BaseModalProcessor):
    def __init__(
        self, tag: str = "audio", audio_processor: AutoFeatureExtractor = None
    ):
        """
        Args:
            tag: 标签，默认 'audio'
            audio_processor: 音频编码器的特征提取器 (如 AutoFeatureExtractor)
        """
        super().__init__(tag)
        self.feature_extractor = audio_processor
        self.target_sampling_rate = self.feature_extractor.sampling_rate
        self.pad_value = self.feature_extractor.padding_value

    def process(self, content: str) -> Dict[str, Any]:
        """
        content: 音频文件路径，例如 "/data/audio/sample_1.wav"
        """
        audio_path = content.strip()

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio: {audio_path}, {e}")
            return {"audio_values": None}

        # 重采样到模型需要的采样率
        if sr != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
            waveform = resampler(waveform)

        # 如果是立体声，通常转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
        )

        # 统一返回 Key
        result = {}
        if "input_features" in inputs:
            result["audio_values"] = inputs["input_features"].squeeze(0)  # (Freq, Time)
        elif "input_values" in inputs:
            result["audio_values"] = inputs["input_values"].squeeze(
                0
            )  # (Sequence_length,)

        result["audio_lens"] = result["audio_values"].shape[-1]

        return result

    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对音频特征进行padding和mask生成
        Args:
            features: 包含processed音频特征的列表
        Returns:
            Tuple[padded_features, attention_mask]
        """
        # 过滤掉None值
        audio_values = [
            f["audio_values"] for f in features if f.get("audio_values") is not None
        ]

        if not audio_values:
            return torch.empty(0), torch.empty(0)

        processed_audio = []
        audio_masks = []

        for wav in audio_values:
            # (Seq, ...)
            if wav.dim() == 2 and wav.shape[0] < wav.shape[1]:
                # (Freq, Seq) -> (Seq, Freq)
                wav = wav.transpose(0, 1)

            processed_audio.append(wav)

            seq_len = wav.shape[0]
            audio_masks.append(torch.ones(seq_len, dtype=torch.long))

        # shape: (B, Seq, Freq) 或 (B, Seq)
        padded_features = pad_sequence(
            processed_audio, batch_first=True, padding_value=self.pad_value
        )
        attention_mask = pad_sequence(audio_masks, batch_first=True, padding_value=0)

        return padded_features, attention_mask

    def encode(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        outputs = encoder(input_features=values, attention_mask=attention_mask)
        features = projector(outputs.last_hidden_state)

        if attention_mask is not None:
            input_lens = attention_mask.sum(dim=1)
            scale = features.shape[1] / values.shape[1]
            valid_out_lens = (
                (input_lens * scale).long().clamp(min=1, max=features.shape[1])
            )
            return [features[i, : valid_out_lens[i]] for i in range(len(features))]

        return [f for f in features]
