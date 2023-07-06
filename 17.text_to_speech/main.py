#pip install TTS
#sudo apt install ffmpeg                 or          https://github.com/markus-perl/ffmpeg-build-script#:~:text=maintain%20different%20systems.-,Installation,-Quick%20install%20and

#KMP_DUPLICATE_LIB_OK=TRUE python main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re

from TTS.api import TTS
from pprint import pprint

#pip install pydub
# from pydub import AudioSegment
# from pydub.playback import play

from auto_everything.terminal import Terminal
terminal = Terminal()

import torch
use_gpu = True if torch.cuda.is_available() else False

pprint(TTS.list_models())

tts_en = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=use_gpu)
#tts_en = TTS("tts_models/en/ljspeech/fast_pitch", gpu=use_gpu)
tts_cn = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST", gpu=use_gpu)


def _speak_it(language: str, text: str):
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "output.wav"))

    if (language == "en"):
        tts = tts_en
    else:
        tts = tts_cn

    if tts.speakers == None:
        tts.tts_to_file(text=text, file_path=output_file)
    else:
        tts.tts_to_file(text=text, file_path=output_file, speaker=tts.speakers[0], language=tts.languages[0], speed=2.5)

    # terminal.run(f"""
     #vlc -I dummy "{output_file}" "vlc://quit"
    # """)

    os.system(f"""
    ffplay -autoexit -nodisp "{output_file}"
              """)

    # os.system(f"""
    # 'c:/Program Files (x86)/VideoLAN/VLC/vlc.exe' -I dummy "{output_file}" "vlc://quit"
    #           """)

    # audio = AudioSegment.from_file(output_file)
    # audio.frame_width = 24000
    # play(audio)


def language_splitor(text: str):
    language_list = []
    index = 0
    while True:
        temp_string = ""
        if (index >= len(text)):
            break
        char = text[index]
        while ord(char) < 128:
            # english
            char = text[index]
            temp_string += char
            index += 1
            if (index >= len(text)):
                break
        if (temp_string.strip() != ""):
            temp_string = temp_string[:-1]
            index -= 1
            language_list.append({
                "language": "en",
                "text": temp_string
            })

        temp_string = ""
        if (index >= len(text)):
            break
        char = text[index]
        while not ord(char) < 128:
            # chinese 
            char = text[index]
            temp_string += char
            index += 1
            if (index >= len(text)):
                break
        if (temp_string.strip() != ""):
            temp_string = temp_string[:-1]
            index -= 1
            language_list.append({
                "language": "cn",
                "text": temp_string
            })

        if (index+1 >= len(text)):
            break

    if len(language_list) > 0:
        language_list[-1]["text"] += text[-1]
    
    new_list = []
    for index, one in enumerate(language_list):
        new_text = language_list[index]["text"].strip()
        if len(new_text) > 0:
            new_list.append({
                'language': one['language'],
                'text': new_text
            })

    return new_list


def speak_it(text: str):
    data_ = language_splitor(text)
    for one in data_:
        print(one)
        _speak_it(language=one["language"], text=one["text"])


text = "Hello, yingshaoxo! I love you so much!"
while True:
    speak_it(text)

    text = input("\n\nWhat you want to say?\n")
