from typing import Any
from pprint import pprint
import json
import re

from auto_everything.io import IO
from auto_everything.disk import Disk, Store
from auto_everything.ml import ML
from auto_everything.string_ import String
ml = ML()
disk = Disk()
string_ = String()
io_ = IO()

all_words_list = []
files = disk.get_files("./data", type_limiter=[".txt"])
for file in files:
    lines = io_.read(file).split("""__**__**__yingshaoxo_is_the_top_one__**__**__""")
    lines = [one.strip() for one in lines if one.strip() != "" and one.isascii()]
    all_words_list += lines
all_words_list = list(set(all_words_list))
all_words_list.sort(key=len, reverse=True)
all_sentence_list = all_words_list

lines = io_.read("./500_basic_words.txt").split("\n")
lines = [line.strip() for line in lines if line.strip() != ""]
words = lines

accept_lines = []
for one in all_sentence_list:
    for word in words:
        if " " + word + " " in one:
            accept_lines.append(one)
            break

global_dict = {}
for line in accept_lines:
    for word in words:
        result = re.findall("[a-zA-Z']+ "+word+" [a-zA-Z']+", line)
        result += re.findall("[a-zA-Z']+ "+word+" ", line)
        result += re.findall(" "+word+" [a-zA-Z']+", line)
        result = [one.strip() for one in result]
        result = list(set(result))
        if len(result) > 0:
            #print(result)
            for each in result:
                if each in global_dict.keys():
                    global_dict[each] += 1
                else:
                    global_dict[each] = 1

final_result = []
for key, value in global_dict.items():
    if value >= 3:
        final_result.append(key)
final_result.sort(key=len, reverse=True)

text = ""
yingshaoxo_translator = ml.Yingshaoxo_Translator()
for one in final_result:
    value = yingshaoxo_translator.english_to_chinese(one)
    if value.endswith("A"):
        value = value[:-1] + "一个"
    print(one, ":", value)
    text += one + "\n" + value
    text += "\n\n_\n\n"
text = text.strip()

io_.write("./additional_en_to_zh_dict_yingshaoxo_version.txt", text)
print("Done")
