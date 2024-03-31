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

cmn_text = io_.read("./cmn.txt")
text_list = cmn_text.split("\n")
text_list = [one.split("	")[0] for one in text_list]
the_regex_dict = text_transformer.get_regex_expression_version_string_dict("\n".join(text_list))
pprint(the_regex_dict)

while True:
    input_text = input("What you want to translate (cn->en)? ")
    output_text = text_transformer.yingshaoxo_regex_expression_based_transformer(input_text, the_regex_dict)
    #output_text = text_transformer.yingshaoxo_regex_expression_based_recursive_transformer(input_text, the_regex_dict)
    print("\n\n----------\n\n")
    print(output_text)
    print("\n\n----------\n\n")
