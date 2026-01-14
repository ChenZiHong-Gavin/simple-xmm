import torch
import torchaudio
from transformers import AutoFeatureExtractor
from typing import Dict, Any
from .base import BaseModalProcessor


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

        # 重采样 (Resample) 到模型需要的采样率
        if sr != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
            waveform = resampler(waveform)

        # 处理多声道：如果是立体声，通常转为单声道
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
