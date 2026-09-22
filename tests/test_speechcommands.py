"""Small no-download check for the Speech Commands feature frontend."""
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torchaudio.transforms import MelSpectrogram

from data import _load_pcm16_wave, _speechcommand_logmel, apply_virtual_shift


def test_speechcommand_logmel_shape_and_scale():
    waveform = torch.sin(torch.linspace(0, 440 * 2 * torch.pi, 16000)).unsqueeze(0)
    mel = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=512, n_mels=32)
    feature = _speechcommand_logmel(waveform, 16000, mel)
    assert feature.shape == (1, 32, 32)
    assert torch.isfinite(feature).all()
    assert abs(float(feature.mean())) < 1e-5


def test_pcm16_wave_loader():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.wav"
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(torch.zeros(160, dtype=torch.int16).numpy().tobytes())
        waveform, sample_rate = _load_pcm16_wave(path)
        assert waveform.shape == (1, 160)
        assert sample_rate == 16000


def test_virtual_shift_preserves_speech_feature_shape():
    feature = torch.randn(1, 32, 32)
    shifted = apply_virtual_shift(feature, 2.0)
    assert shifted.shape == feature.shape
    assert torch.equal(shifted, feature + 2.0)


if __name__ == "__main__":
    test_speechcommand_logmel_shape_and_scale()
    test_pcm16_wave_loader()
    test_virtual_shift_preserves_speech_feature_shape()
    print("SPEECHCOMMANDS_FRONTEND_OK")
