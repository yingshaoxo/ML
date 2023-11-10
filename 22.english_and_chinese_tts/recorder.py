"""
sudo apt install portaudio19-dev python3-pyaudio
python3 -m pip install pyaudio
conda install -c anaconda pyaudio

python3 -m pip install lameenc

python3 -m pip install keyborad
"""
import pyaudio
import wave
import io
import lameenc

import sys
import select
import tty
import termios
import os
import re

from rnnoise_wrapper import RNNoise
denoiser = RNNoise()

import pydub

from auto_everything.io import IO
from auto_everything.terminal import Terminal
from auto_everything.ml import ML
from auto_everything.disk import Disk, Store
io_ = IO()
terminal = Terminal()
ml = ML()
disk = Disk()
store = Store("yingshaoxo_tts_data_record")
text_preprocessor = ml.Yingshaoxo_Text_Preprocessor()

audio_folder_path = "./voices"
disk.create_a_folder(audio_folder_path)

class NonBlockingConsole():
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False

    def is_esc_pressed(self):
        if self.get_data() == '\x1b':  # x1b is ESC
            return True
        else:
            return False

def get_input_key():
    sys.stdin.buffer.read(1)
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        b = os.read(sys.stdin.fileno(), 3).decode()
        if len(b) == 3:
            k = ord(b[2])
        else:
            k = ord(b)
        key_mapping = {
            127: 'backspace',
            10: 'return',
            32: 'space',
            9: 'tab',
            27: 'esc',
            65: 'up',
            66: 'down',
            67: 'right',
            68: 'left'
        }
        return key_mapping.get(k, chr(k))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def record_wav(filename, framerate=44100):
    # Create a PyAudio object.
    p = pyaudio.PyAudio()

    # Open a stream to record audio.
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=framerate, input=True)

    # Create a NumPy array to store the audio data.
    frames = []

    # Start recording.
    print("Recording... Press ctrl+c to stop recording.")
    try:
        with NonBlockingConsole() as no_blocking:
            while True:
                # Check if the user has hit the escape button.
                data = stream.read(1024)
                frames.append(data)

                if no_blocking.is_esc_pressed():
                    print("Stopped by esc key.")
                    break
    except KeyboardInterrupt:
        print("Stopped by ctrl+c keyborad.")

    # Stop the stream and close the PyAudio object.
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the NumPy array to a WAV file.
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(framerate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Convert the WAV file to an MP3 file.
    #convert_wav_to_mp3(filename, filename)

def play_audio(path: str):
    terminal.run_command(f"""
        vlc -I dummy "{path}" "vlc://quit"
    """)

def reduce_noise_in_audio(source_path: str, target_path: str):
    audio = denoiser.read_wav(source_path)
    denoised_audio = denoiser.filter(audio)
    denoiser.write_wav(target_path, denoised_audio)

def remove_start_and_end_silence(input_wav_file, output_wav_file, silence_threshold=-55):
    # Load the WAV file.
    audio_segment = pydub.AudioSegment.from_wav(input_wav_file)

    # Detect the start and end of the silence.
    start_silence_len = pydub.silence.detect_leading_silence(audio_segment, silence_threshold)
    end_silence_len = pydub.silence.detect_leading_silence(audio_segment.reverse(), silence_threshold)

    # Trim the silence from the beginning and end of the audio segment.
    audio_segment = audio_segment[start_silence_len:-end_silence_len]

    # Export the trimmed audio to a new file
    audio_segment.export(output_wav_file, format="wav")

def convert_wav_to_mp3(input_wav_file, output_mp3_file):
    # Read the input WAV file
    with wave.open(input_wav_file, 'rb') as wav:
        # Get the waveform data
        framerate = wav.getframerate()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        pcm_data = wav.readframes(wav.getnframes())

    # Create a BytesIO buffer for MP3 output
    mp3_buffer = io.BytesIO()

    # Create a LameEncoder instance
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(320)  # Set the desired bitrate (e.g., 320 kbps)
    encoder.set_in_sample_rate(framerate)
    encoder.set_channels(num_channels)
    encoder.set_quality(2)  # Set the quality level

    # Encode the PCM data to MP3
    mp3_buffer = encoder.encode(pcm_data)
    mp3_buffer += encoder.flush()

    # Write the encoded MP3 data to the output file
    with open(output_mp3_file, 'wb') as mp3_file:
        mp3_file.write(mp3_buffer)


def find_longest_sub_sentence_to_audio_dict(use_mp3: bool = False) -> dict[str, str]:
    if use_mp3 == True:
        type_limiter = [".mp3"]
    else:
        type_limiter = [".wav"]

    the_dict = {}
    files = disk.get_files(audio_folder_path, recursive=True, type_limiter=type_limiter, use_gitignore_file=True)
    for file in files:
        files_name, _ = disk.get_stem_and_suffix_of_a_file(file)
        files_name = files_name.lower()
        the_dict[files_name] = file
    return the_dict

def play_with_could_get_played(input_text: str, the_dict: dict[str, str], without_play: bool = False) -> list[str]:
    new_text_audio_needed_list = []

    input_text = input_text.lower()
    sub_sentence_list = text_preprocessor.string_split_to_pure_sub_sentence_segment_list(input_text, without_punctuation=True, without_number=False, not_include_punctuations=" _-'%")
    for sub_sentence in sub_sentence_list:
        found = True
        while found == True and sub_sentence.strip() != "":
            sub_sentence = sub_sentence.strip()
            found = False
            for key, value in the_dict.items():
                if sub_sentence.startswith(key):
                    if without_play == False:
                        play_audio(value)
                    sub_sentence = sub_sentence.replace(key, "", 1)
                    found = True
                    break
            if found == False and sub_sentence.strip() != "":
                new_text_audio_needed_list.append(sub_sentence.strip())
                break

    return new_text_audio_needed_list


#store.set("index", 0)
#exit()

Play_Audio = False

source_text = io_.read("yingshaoxo_telegram_data_2023_11_10.txt")
source_text = re.sub(r'http\S+', '', source_text)
source_text_list = source_text.split("\n\n_____________________\n\n")
source_text_list = [one.strip() for one in source_text_list if one.strip() != ""]
source_text_list = list(reversed(source_text_list))
os.system("clear")
while True:
    #input_text = input("What you want to say? ")
    index = store.get("index", 0)
    input_text = source_text_list[index]

    sub_sentence_to_audio_dict = find_longest_sub_sentence_to_audio_dict()

    new_text_audio_needed_list = play_with_could_get_played(input_text, sub_sentence_to_audio_dict, not Play_Audio)
    print("You need to record audios for: ", new_text_audio_needed_list)
    print("\n\n")

    repeat = False
    for new_text in new_text_audio_needed_list:
        print("You need to record audio for:    ", new_text)
        response = input(f"Press enter to start the recording.")

        temp_record_path = disk.join_paths(audio_folder_path, f".{new_text}.wav")
        record_wav(temp_record_path)

        remove_noise_version_path = disk.join_paths(audio_folder_path, f"{new_text}.wav")
        reduce_noise_in_audio(temp_record_path, remove_noise_version_path)

        temp_record_path = disk.join_paths(audio_folder_path, f".{new_text}.wav")
        remove_start_and_end_silence(remove_noise_version_path, temp_record_path)
        disk.move_a_file(temp_record_path, remove_noise_version_path)

        disk.delete_a_file(temp_record_path)
        play_audio(remove_noise_version_path)

        os.system("clear")
        if disk.get_file_size(remove_noise_version_path) == 44:
            disk.delete_a_file(remove_noise_version_path)
            repeat = True
            break
        input("Ready for the next record? ")

    if repeat == True:
        continue

    store.set("index", index + 1)
    input("Ready for the next text? ")

"""
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
"""
