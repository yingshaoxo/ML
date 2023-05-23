from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["test_voice.wav"]

transcriptions = model.transcribe(audio_paths)
print(transcriptions[0]['transcription'])