from typing import Any
from pprint import pprint
import json

from auto_everything.io import IO
from auto_everything.disk import Disk, Store
from auto_everything.ml import ML
from auto_everything.string_ import String
ml = ML()
disk = Disk()
string_ = String()
io_ = IO()

all_words_list = []
files = disk.get_files("./data", type_limiter=[".json"])
for file in files:
    all_words_list += json.loads(io_.read(file))
all_words_list = list(set(all_words_list))
all_words_list.sort(key=len, reverse=True)

#the_global_dict = {}
text = ""
yingshaoxo_translator = ml.Yingshaoxo_Translator()
for one in all_words_list:
    value = yingshaoxo_translator.english_to_chinese(one)
    print(one, ":", value)
    text += one + "\n" + value
    text += "\n\n_\n\n"
text = text.strip()

#io_.write("./en_to_zh_word_dict_yingshaoxo_version.json", json.dumps(the_global_dict, indent=4, ensure_ascii=False))
io_.write("./en_to_zh_word_dict_yingshaoxo_version.txt", text)
print("Done")
