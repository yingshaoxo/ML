import random
import re

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
text_transformer = ml.Yingshaoxo_Text_Transformer()

input_text_list = []
output_text_list = []
text = io_.read("/home/yingshaoxo/CS/ML/13.seq2seq/cmn.txt")
for one in text.split("\n"):
    splits = one.split("	")
    if len(splits) == 2:
        en, cn = splits
        input_text_list.append(en)
        output_text_list.append(cn)

result = string_.get_meaning_group_dict_in_text_list(input_text_list)
print(result)
print(len(result))
exit()

