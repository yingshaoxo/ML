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


def get_similarity_score_of_two_sentence_by_position_match(sentence1: str, sentence2: str) -> float:
    sentence1_length = len(sentence1)
    sentence2_length = len(sentence2)
    counting = 0
    min_length = min(sentence1_length, sentence2_length)
    for index in range(min_length):
        char = sentence1[index]
        another_sentence_char = sentence2[index]
        if char == another_sentence_char:
            counting += 1
    return counting / min_length


yingshaoxo_translator = ml.Yingshaoxo_Translator()


def get_regex_version_of_keys_and_values(a_dict: dict[str, str]) -> dict[str, str] | None:
    """
    a_dict = {
        "That is steven, my uncle.": "I see, steven is your uncle.",
        "That is wind_god, my uncle.": "I see, wind_god is your uncle.",
    }

    it will return:
        {
            "That is (?P<name>.*?), my uncle.": "I see, {name} is your uncle.",
        }
    """
    items = list(a_dict.items())
    items.sort(key=lambda x: len(x[0]))

    base_sentence = items[0]
    another_sentence = items[1]

    if len(base_sentence) == 0 or len(another_sentence) == 0:
        return None

    if base_sentence[0] == another_sentence[0]:
        return {base_sentence[0]: base_sentence[1]}

    start_index = 0
    for index, char in enumerate(base_sentence[0]):
        another_sentence_char = another_sentence[0][index]
        if char == another_sentence_char:
            pass
        else:
            start_index = index
            break

    end_index = len(base_sentence[0])
    for index, char in enumerate(reversed(another_sentence[0])):
        if abs(-index-1) == len(base_sentence[0]):
            break
        another_sentence_char = base_sentence[0][-index-1]
        if char == another_sentence_char:
            pass
        else:
            index = len(base_sentence[0]) - index
            end_index = index
            break

    the_fuzz_keyword = base_sentence[0][start_index: end_index]
    if len(the_fuzz_keyword) == 0:
        return a_dict

    start_part = base_sentence[0][:start_index]
    end_part = base_sentence[0][end_index:]
    if (not start_part.endswith(" ")):
        return None
    if (not end_part.startswith(" ")):
        if len(end_part) != 1:
            return None
        if len(end_part) == 1:
            if end_part[-1] not in ",.!":
                return None

    translated = yingshaoxo_translator.english_to_chinese(start_part + "yingshaoxo" + end_part)
    if "Yingshaoxo" not in translated:
        return None
    translated = translated.replace("Yingshaoxo", "{}")

    return {
        re.escape(start_part) + f"(.*?)" + re.escape(end_part):
            translated
    }


def find_regex_match_string(input_text: str, regex_expression_dict: dict[str, str]) -> str | None:
    regex_keys = sorted(list(regex_expression_dict.keys()), key=len, reverse=True)

    def the_transformer(input_text: str) -> str:
        for key in regex_keys:
            result = re.match(key, input_text, flags=re.DOTALL)
            if result != None:
                if len(result.groups()) < 1:
                    # no regex inside of that dict
                    #return regex_expression_dict[key]
                    continue

                value = result.group(1)
                return value
        return None

    return the_transformer(input_text)


input_text_list = []
output_text_list = []
text = io_.read("/home/yingshaoxo/CS/ML/13.seq2seq/cmn.txt")
for one in text.split("\n"):
    splits = one.split("	")
    if len(splits) == 2:
        en, cn = splits
        input_text_list.append(en)
        output_text_list.append(cn)

#input_text_list = [
#"What do you think?",
#"What do you like to eat?",
#"like to eat?",
#"like to take?",
#"like to go?",
#]

global_dict = {}
window_size = 2

queue = input_text_list
while len(queue) > 0:
    for index in range(len(queue)):
        if index < window_size:
            continue
        sub_input_list = queue[index-window_size:index]

        temp_dict = {}
        for a in sub_input_list:
            temp_dict.update({a: a})
        if len(temp_dict.keys()) >= 2:
            #print(temp_dict)
            result = get_regex_version_of_keys_and_values(temp_dict)
            if result != None:
                key = list(result.keys())[0]
                if key not in global_dict.keys():
                    global_dict.update(result)
                    print(result)

    """
    new_queue = []
    for element in queue:
        result = find_regex_match_string(element, global_dict)
        if result != None:
            print(result)
            if result.strip() != "":
                new_queue.append(result)
    new_queue = list(set(new_queue))
    new_queue.sort(key=len, reverse=True)
    queue = new_queue
    """
    break

pprint(global_dict)
print(len(global_dict.keys()))
#io_.write("./test.json", json.dumps(global_dict, indent=4, ensure_ascii=False))
io_.write("./test2.json", json.dumps(global_dict, indent=4, ensure_ascii=False))
