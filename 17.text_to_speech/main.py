#pip install TTS
#pip install pydub
#sudo apt install ffmpeg                 or          https://github.com/markus-perl/ffmpeg-build-script#:~:text=maintain%20different%20systems.-,Installation,-Quick%20install%20and
from TTS.api import TTS
from pprint import pprint

from pydub import AudioSegment
from pydub.playback import play

from auto_everything.terminal import Terminal
terminal = Terminal()

import os

pprint(TTS.list_models())

tts = TTS("tts_models/en/ljspeech/fast_pitch")

print(tts.speakers)
print(tts.languages)

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "output.wav"))

text = "Hello, yingshaoxo! This is a test!"
while True:
    tts.tts_to_file(text=text, file_path=output_file)

    # terminal.run(f"""
    # vlc -I dummy "{output_file}" "vlc://quit"
    # """)

    audio = AudioSegment.from_file(output_file)
    audio.frame_width = 24000
    play(audio)

    text = input("\n\nWhat you want to say?\n")