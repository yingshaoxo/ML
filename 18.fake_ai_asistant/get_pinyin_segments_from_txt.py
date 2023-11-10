from auto_everything.disk import Disk
from auto_everything.io import IO
disk = Disk()
io_ = IO()

import json
import jieba

if disk.exists("./words.json"):
    words_data_list = json.loads(io_.read("./words.json"))
else:
    files = disk.get_files("./input_txt_files")
    files += ["/home/yingshaoxo/Downloads/鬼吹灯.txt"]

    print(files)

    text = ""
    for file in files:
        text += io_.read(file)

    words_data_list = list(set(list(jieba.cut(text))))

    io_.write('./words.json', json.dumps(words_data_list, sort_keys=True)) 

from xpinyin import Pinyin
pinyin = Pinyin()

words_data_list = list(set(list("".join(words_data_list))))
a_set = set()
for word in words_data_list:
    #pinyin_text = pinyin.get_pinyin(word, tone_marks='marks')
    pinyin_text = pinyin.get_pinyin(word)
    #print(word, pinyin_text)
    a_set.add(pinyin_text)

print(len(a_set))