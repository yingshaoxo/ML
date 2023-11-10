import random
from auto_everything.ml import ML
from auto_everything.disk import Disk
ml = ML()
disk = Disk()

text_generator = ml.Yingshaoxo_Text_Generator()
text_preprocessor = ml.Yingshaoxo_Text_Preprocessor()
text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files", type_limiter=[".txt"])
text = text.replace("__**__**__yingshaoxo_is_the_top_one__**__**__", "")
#text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="/home/yingshaoxo/Downloads/source_txt", type_limiter=[".txt"])


def get_word_list_from_string(text: str, include_punctuation: bool = False) -> list[str]:
    final_list = []

    sentence_segment_list = text_preprocessor.split_string_into_list_by_punctuations(text, not_include_punctuations="\n' _")
    for segment in sentence_segment_list:
        if segment["language"] == "punctuation":
            if include_punctuation == False:
                continue
            else:
                final_list += list(segment["text"])
                continue

        segment = segment["text"].strip()
        if segment == "":
            continue

        segment_list = []
        if text_preprocessor.is_english_string(segment):
            segment_list += segment.split(" ")
        else:
            segment_list += list(segment)

        segment_list = [one.strip() for one in segment_list]
        final_list += segment_list

    return final_list

def get_line_sort_list(source_text_data: str):
    """
    source_text = "hi you\n yingshaoxo"
    It returns:
    [
        [["hi", "you"], ["yingshaoxo"]],
    ]
    """
    follow_lines = 3

    line_sort_list = []
    lines = source_text_data.split("\n")
    lines = [line for line in lines if line.strip() != ""]
    for index, line in enumerate(lines):
        line = line.strip()

        if index >= len(lines) - follow_lines:
            continue

        word_list = get_word_list_from_string(line, include_punctuation=True)
        word_list = list(set(word_list))

        next_line_word_list = []
        for i in range(1, follow_lines+1):
            next_line = lines[index+i].strip()
            next_line_word_list += get_word_list_from_string(next_line, include_punctuation=True)
        next_line_word_list = list(set(next_line_word_list))

        line_sort_list.append(
            [word_list, next_line_word_list]
        )

    return line_sort_list

def sort_sentence_in_text(input_text: str, line_sort_list: list, sort_times: int = 1, no_additional_new_line: bool = True) -> tuple[str, bool]:
    def do_the_work(input_text):
        input_lines = input_text.split("\n")
        input_lines = [line for line in input_lines if line.strip() != ""]
        for index, line in enumerate(input_lines):
            word_list = get_word_list_from_string(line, include_punctuation=True)
            word_list = list(set(word_list))
            for line_data in line_sort_list:
                all_in = True
                for word in word_list:
                    if word not in line_data[0]:
                        all_in = False
                        break
                if all_in == False:
                    continue
                elif all_in == True:
                    previous_lines = input_lines[:index]
                    for previous_index, previous_line in enumerate(previous_lines):
                        previous_line_word_list = get_word_list_from_string(previous_line, include_punctuation=True)
                        previous_line_word_list = list(set(previous_line_word_list))
                        all_in_value = True
                        for word in previous_line_word_list:
                            if word not in line_data[1]:
                                all_in_value = False
                                break
                        if all_in_value==True:
                            if no_additional_new_line == True:
                                input_lines.insert(index+1, previous_line)
                                del input_lines[previous_index]
                                return False, '\n'.join(input_lines)
                            else:
                                if index >= len(input_lines) - 1:
                                    input_text = input_text.replace(previous_line + "\n", "", 1)
                                    input_text = input_text + "\n" + previous_line
                                else:
                                    next_line = input_lines[index+1]
                                    input_text = input_text.replace(previous_line + "\n", "", 1)
                                    input_text = input_text.replace(next_line, previous_line +"\n"+ next_line, 1)
                                return False, input_text
        return True, input_text

    done = False
    new_text = input_text
    for _ in range(sort_times):
        done, new_text = do_the_work(new_text)

    return new_text, done

def sort_sub_sentence_in_text(input_text: str, source_text: str) -> list[str]:
    sub_sentence_sort_list = text_preprocessor.string_split_to_pure_sub_sentence_segment_list(source_text, without_punctuation=True)
    input_text_sub_sentence_list = text_preprocessor.string_split_to_pure_sub_sentence_segment_list(input_text, without_punctuation=True)

    def _sort_by_source_order_unknown(input_list, source_order_list):
        """Sorts the input_list by the source_order_list order, and keep unknown elements in input_list order untouched."""

        # Create a dictionary mapping each element in source_order_list to its index.
        element_to_index = {element: i for i, element in enumerate(source_order_list)}

        # Create a list of known elements and a list of unknown elements.
        known_elements = []
        unknown_elements = []
        for element in input_list:
            if element in element_to_index:
                known_elements.append(element)
            else:
                unknown_elements.append(element)

        # Sort the known elements using the dictionary as a key.
        sorted_known_elements = sorted(known_elements, key=lambda element: element_to_index[element])

        # Combine the sorted known elements and the unknown elements.
        sorted_list = sorted_known_elements + unknown_elements
        return sorted_list

    input_text_sub_sentence_list = _sort_by_source_order_unknown(input_text_sub_sentence_list, sub_sentence_sort_list)

    return input_text_sub_sentence_list

input_text = """
I'm always 18 years old.

Then how old you are?


如何反抗集权政府？


1. 做好准备


个体行为:

2. 隐藏反叛思想，暗中捣乱，像穿了隐身衣一样战斗
2. 宣传新思想、新文化、号召新行动

1. 不听宣传、不信宣传、不执行指令

全员行为:
"""

result = sort_sub_sentence_in_text(input_text, text)
input_text = "\n".join([one for one in result])

#line_sort_list = get_line_sort_list(text)
#input_text, done = sort_sentence_in_text(input_text=input_text, line_sort_list=line_sort_list, sort_times=5)

input_text = text_generator.correct_sentence_by_using_yingshaoxo_regex_method(input_text=input_text, source_data_text=text, level=3)
print(input_text)
