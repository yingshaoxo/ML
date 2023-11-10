# yingshaoxo TTS

```
TTS hard coding method 2:


1. Text map to 64k mp3 audio, play longest substring first

2. Use ",." symbol to separate text, so you get less repeated text data

3. When you got 1GB of data, you get a well functioned TTS

> You could even use speech recognition to collect audio to text dict data.

> By using this method, you could get almost 100% accurate TTS for your voice
```

If you want to have a better results, you have to cut the start_space(silence) and end_space(silence) for each audio you have, and reduce the noises in your audios.

## Use recorder.py to collect human voice
Here I use yingshaoxo diary data with my own recording. (I do the record one by one)

All data is my data, so it can be MIT licensed.

## Use data_filling.py to fill missing words or char voice automatically by using voice recognizion tech
I may use vosk.
