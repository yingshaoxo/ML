#pip install TTS
#pip install pydub
#sudo apt install ffmpeg                 or          https://github.com/markus-perl/ffmpeg-build-script#:~:text=maintain%20different%20systems.-,Installation,-Quick%20install%20and

#KMP_DUPLICATE_LIB_OK=TRUE python main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from TTS.api import TTS
from pprint import pprint

from auto_everything.terminal import Terminal
terminal = Terminal()

import re

#tts = TTS("tts_models/en/ljspeech/fast_pitch")
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "output_audio.wav"))

def say_somthing(text: str):
    text = re.sub(r'[^\x00-\x7F]', '', text)
    try:
        tts.tts_to_file(text=text, file_path=output_file)

        os.system(f"""
        ffplay -autoexit -nodisp "{output_file}"
                """)
    except Exception as e:
        print(e)