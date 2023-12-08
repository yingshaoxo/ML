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

import jieba


text_generator = ml.Yingshaoxo_Text_Generator()
text_transformer = ml.Yingshaoxo_Text_Transformer()


the_regex_dict = {}
text = io_.read("./en_to_zh_word_dict_yingshaoxo_version.txt")
data_list = [one for one in text.split("\n\n_\n\n") if one != ""]
for one in data_list:
    value, key = one.split("\n")[:2]
    the_regex_dict[key] = value

text = io_.read("./additional_en_to_zh_dict_yingshaoxo_version.txt")
data_list = [one for one in text.split("\n\n_\n\n") if one != ""]
for one in data_list:
    key, value = one.split("\n")[:2]
    the_regex_dict[key] = value

for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ',.:":
    if char not in the_regex_dict.keys():
        the_regex_dict[char] = char


while True:
    input_text = input("What you want to translate (cn->en)? ")
    output_text = text_transformer.pure_string_dict_based_sequence_transformer(" ".join(jieba.cut(input_text.lower())), the_regex_dict, add_space=True)
    #output_text = text_transformer.yingshaoxo_regex_expression_based_recursive_transformer(input_text, the_regex_dict)
    print("\n\n----------\n\n")
    print(output_text)
    print("\n\n----------\n\n")
