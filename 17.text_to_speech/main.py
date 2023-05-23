import torch
import torchaudio

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

text = "How to speak Chinese?"

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)

torchaudio.save("output_audio.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)

# waveglow = torch.hub.load(
#     "NVIDIA/DeepLearningExamples:torchhub",
#     "nvidia_waveglow",
#     model_math="fp32",
#     pretrained=False,
# )
# checkpoint = torch.hub.load_state_dict_from_url(
#     "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
#     progress=False,
#     map_location=device,
# )
# state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

# waveglow.load_state_dict(state_dict)
# waveglow = waveglow.remove_weightnorm(waveglow)
# waveglow = waveglow.to(device)
# waveglow.eval()

# with torch.no_grad():
#     waveforms = waveglow.infer(spec)

# torchaudio.save("output_audio.wav", waveforms[0:1].cpu(), sample_rate=22050)