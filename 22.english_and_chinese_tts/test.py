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

input_text = """
我这么给你讲吧！

邪恶势力不仅要控制终端用户 (通过应用商店，只上架他们允许上架的。通过控制舆论、搜索引擎，删除掉一切他们不允许的)

邪恶势力还要控制程序开发者，比如提供阉割版SDK、限制开发者权限的使用。又比如提供阉割版的自动写代码软件，只能自动生成被审查后的低端代码。

#控制用户 #控制开发者 #yingshaoxo
"""

result = text_preprocessor.string_split_to_pure_sub_sentence_segment_list(input_text, without_punctuation=True, without_number=True, not_include_punctuations=" _-'")
print(result)
