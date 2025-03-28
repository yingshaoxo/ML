#import random
#
#def get_similarity_score_of_two_sentence_by_position_match(sentence1: str, sentence2: str) -> float:
#    sentence1_length = len(sentence1)
#    sentence2_length = len(sentence2)
#    base_sentence = None
#    another_sentence = None
#    if sentence1_length <= sentence2_length:
#        base_sentence = sentence1
#        another_sentence = sentence2
#    else:
#        base_sentence = sentence2
#        another_sentence = sentence1
#
#    counting = 0
#    for index, char in enumerate(base_sentence):
#        another_sentence_char = another_sentence[index]
#        if char == another_sentence_char:
#            counting += 1
#
#    return counting / len(base_sentence)
#
#
#def get_random_number_string(length: int) -> str:
#    return ''.join(random.choice("0123456789") for _ in range(length))
#
#
#def get_regex_version_of_keys_and_values(a_dict: dict[str, str]) -> dict[str, str]:
#    """
#    a_dict = {
#        "That is steven, my uncle.": "I see, steven is your uncle.",
#        "That is wind_god, my uncle.": "I see, wind_god is your uncle.",
#    }
#
#    it will return:
#        {
#            "That is (?P<name>.*?), my uncle.": "I see, {name} is your uncle.",
#        }
#    """
#    items = list(a_dict.items())
#    items.sort(key=lambda x: len(x[0]))
#
#    base_sentence = items[0]
#    another_sentence = items[1]
#
#    if base_sentence[0] == another_sentence[0]:
#        return {base_sentence[0]: base_sentence[1]}
#
#    start_index = 0
#    for index, char in enumerate(base_sentence[0]):
#        another_sentence_char = another_sentence[0][index]
#        if char == another_sentence_char:
#            pass
#        else:
#            start_index = index
#            break
#
#    end_index = len(base_sentence[0])
#    for index, char in enumerate(reversed(another_sentence[0])):
#        another_sentence_char = base_sentence[0][-index-1]
#        if char == another_sentence_char:
#            pass
#        else:
#            index = len(base_sentence[0]) - index
#            end_index = index
#            break
#
#    the_fuzz_keyword = base_sentence[0][start_index: end_index]
#    if len(the_fuzz_keyword) == 0:
#        return a_dict
#
#    random_string = get_random_number_string(4)
#    return {
#        base_sentence[0][:start_index] + f"(?P<{random_string}>.*?)" + base_sentence[0][end_index:] :
#            base_sentence[1].replace(base_sentence[0][start_index: end_index], f"{{{random_string}}}")
#    }
#
#result = get_regex_version_of_keys_and_values({
#    "That is steven, my uncle.": "I see, steven is your uncle.",
#    "That is wind_god, my uncle.": "I see, wind_god is your uncle.",
#})
#result = get_regex_version_of_keys_and_values({
#"Did you see ?": "I see AA.",
#"Did you see akj?": "I see CC.",
#})
#print(result)
#exit()

import re
from auto_everything.string_ import String
string_ = String()

def excel_sheet_number_to_column_letter(column_number: int):
    if column_number < 0:
        column_number = 0

    column_number += 1

    result = ''
    while column_number > 0:
        column_number, remainder = divmod(column_number - 1, 26)
        result = chr(65 + remainder) + result

    return result

def get_regex_expression_from_current_text_and_following_text(current_text: str, following_text: str) -> tuple[str, str]:
    sub_string_list = string_.get_all_sub_string(text=current_text)
    sub_string_list.sort(key=len, reverse=True)

    fake_current_text = current_text
    fake_following_text = following_text
    new_current_text = current_text
    new_following_text = following_text
    counting = 0
    for index, sub_string in enumerate(sub_string_list):
        if (sub_string in fake_current_text) and (sub_string in fake_following_text):
            fake_following_text = fake_following_text.replace(sub_string, "")
            new_current_text_list = new_current_text.split(sub_string)
            new_current_text_list = [re.escape(one)  for one in new_current_text_list]
            new_current_text = f"(?P<y{counting}>.*?)".join(new_current_text_list)
            #new_current_text = new_current_text.replace(sub_string, f"(?P<y{counting}>.*?)") # You have to find a way to avoid new sub_string replace old regex expression
            new_following_text = new_following_text.replace(sub_string, f"{{y{counting}}}")
            counting += 1
            break

    return new_current_text, new_following_text

def get_regex_expression_version_string_dict(input_text: str, seporator: str = "\n") -> dict[str, str]:
    final_dict = {}

    text_list = input_text.split(seporator)
    for index, text in enumerate(text_list):
        if index + 1 > len(text_list) - 1:
            break

        text = text.strip()
        next_text = text_list[index+1].strip()
        if text != "" and next_text != "":
            key, value = get_regex_expression_from_current_text_and_following_text(text, next_text)
            #print(key, value)
            final_dict[key] = value

    return final_dict

def yingshaoxo_regex_expression_based_transformer(input_text: str, regex_expression_dict: dict[str, str]) -> str:
    for key in sorted(list(regex_expression_dict.keys()), key=len, reverse=True):
        result = re.search(key, input_text)
        if result != None:
            return regex_expression_dict[key].format(**result.groupdict())
    return ""


result = yingshaoxo_regex_expression_based_transformer(
    input_text="你是漂亮老师吗？",
    regex_expression_dict=get_regex_expression_version_string_dict(input_text="""
你是傻逼吗？
你才是傻逼！
    """)
)
print(result)

result = yingshaoxo_regex_expression_based_transformer(
    input_text="Are you smart? ",
    regex_expression_dict=get_regex_expression_version_string_dict(input_text="""
Are you stupid?
You are stupid!
    """)
)
print(result)

exit()


"""
We will use char level operation to get unknown keywords regex from "multiple key -> one value" data pairs

Hi AA -> Hi you.
Hi BB -> Hi you.
Hi CC -> Hi you.

We need to get "Hi (.*?) -> Hi you." from above data automatically.



Did you see AA? => I see AA.
Did you see BB? => I see BB.

We need to get "Did you see (?P<someone>.*?)? -> I see {someone}." from above data automatically.



That is steven, my uncle. => I see, steven is your uncle.
That is wind_god, my uncle. => I see, wind_god is your uncle.

We need to get "That is (?P<name>.*?), my uncle. => I see, {name} is your uncle." from above data automatically.
"""

"""
Or, you could think it as a very simple problem, if you got current_line of text, if some sub_string appears in the next_line of text, you can safely replace it with regex expression.

For example: what is the age of uncle? => uncle is 18 years old.

You can just do a search for every sub_string in the first sentence, if that substring appears 1 or more times in the second sentence, you get a general regex sentence.

For example: what is the age of (?P<name>.*?)? => {name} is 18 years old.

And later when you meet new input, if it full matchs any regex expression, you return following sentence with related content formated. In other words, you are returning a reactive answer than fixed answer.
"""
