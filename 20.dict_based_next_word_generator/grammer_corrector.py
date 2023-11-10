from auto_everything.ml import ML
from auto_everything.disk import Disk
ml = ML()
disk = Disk()

text_generator =  ml.Yingshaoxo_Text_Generator()

text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files")
#global_string_dict = text_generator.get_global_string_dict_by_using_yingshaoxo_method(source_text_data=text, levels=10)

def get_global_string_corrector_dict_by_using_yingshaoxo_method(source_text_data: str, levels: int = 10, for_minus_character: bool = False):
    global_string_dict = {
    }

    seperator = "☺"

    def get_x_level_dict(source_text: str, x: int):
        level_dict = {}
        for index, _ in enumerate(source_text):
            if index < x:
                continue
            if index == len(source_text) - x:
                break
            if for_minus_character == True:
                current_chars = source_text[index-x: index] + seperator + source_text[index: index+x]
                center_char = ""
            else:
                current_chars = source_text[index-x: index] + seperator + source_text[index+1: index+x+1]
                center_char = source_text[index]
            if current_chars in level_dict:
                if center_char in level_dict[current_chars]:
                    level_dict[current_chars][center_char] += 1
                else:
                    level_dict[current_chars][center_char] = 1
            else:
                level_dict[current_chars] = {center_char: 1}

        pure_level_dict = {}
        for key, value in level_dict.items():
            biggest_value = 0
            biggest_key = None
            for key2, value2 in value.items():
                if value2 > biggest_value:
                    biggest_value = value2
                    biggest_key = key2
            pure_level_dict[key] = biggest_key

        return pure_level_dict

    max_level = levels
    for level in reversed(list(range(1, 1+max_level))):
        global_string_dict[level] = get_x_level_dict(source_text_data, level)
        break

    return global_string_dict

def correct_sentence_by_using_yingshaoxo_method(input_text: str, levels: int = 10, source_text_data: str|None = None, global_string_corrector_dict: dict|None = None, plus_character: bool = False, minus_character: bool = False) -> any:
    """
    This will correct text based on pure text or hash map or hash dict. if you use it in memory, the speed would be super quick.
    If you can modify this to word level, the accuracy could be 100%
    """
    if source_text_data == None:
        #source_text_data = self.text_source_data
        source_text_data = ""

    if global_string_corrector_dict != None:
        pass
    else:
        #global_string_dict = self.get_global_string_dict_by_using_yingshaoxo_method(source_text_data, levels)
        global_string_corrector_dict = {}

    input_text = "\n"*len(global_string_corrector_dict) + input_text + "\n"*len(global_string_corrector_dict)

    seperator = "☺"
    new_text = ""
    for level in global_string_corrector_dict.keys():
        for index, _ in enumerate(input_text):
            if index < (level-1):
                new_text += input_text[index]
                continue
            if index >= len(input_text) - level:
                new_text += input_text[index]
                continue

            if plus_character == True:
                current_chars = input_text[index-level: index] + seperator + input_text[index: index+level]
                if current_chars in global_string_corrector_dict[level].keys():
                    new_text += global_string_corrector_dict[level][current_chars] + input_text[index]
                else:
                    new_text += input_text[index]
            elif minus_character == True:
                current_chars = input_text[index-level: index] + seperator + input_text[index+1: index+1+level]
                if current_chars in global_string_corrector_dict[level].keys():
                    new_text += ""
                else:
                    new_text += input_text[index]
            else:
                current_chars = input_text[index-level: index] + seperator + input_text[index+1: index+1+level]
                if current_chars in global_string_corrector_dict[level].keys():
                    new_text += global_string_corrector_dict[level][current_chars]
                else:
                    new_text += input_text[index]
        break
    return new_text


global_string_corrector_dict_for_adding = get_global_string_corrector_dict_by_using_yingshaoxo_method(text, 3)
global_string_corrector_dict_for_minus = get_global_string_corrector_dict_by_using_yingshaoxo_method(text, 3, for_minus_character=True)
while True:
    input_text = input("What you want to say? ")

    result = correct_sentence_by_using_yingshaoxo_method(input_text, global_string_corrector_dict=global_string_corrector_dict_for_minus, minus_character=True)
    result = correct_sentence_by_using_yingshaoxo_method(result, global_string_corrector_dict=global_string_corrector_dict_for_adding, plus_character=False)
    result = correct_sentence_by_using_yingshaoxo_method(result, global_string_corrector_dict=global_string_corrector_dict_for_adding, plus_character=True)
    print(result)
