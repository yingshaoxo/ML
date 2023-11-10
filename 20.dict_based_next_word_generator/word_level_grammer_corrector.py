from auto_everything.ml import ML
from auto_everything.disk import Disk
ml = ML()
disk = Disk()

def split_string_into_list_by_punctuations(input_text, special_punctuations = "\n ,.!?()[]{}<>;:’‘“”\"'`’‘「」『』【】〖〗《》《 》〈 〉〔 〕（ ）﹙ ﹚【 】［ ］｛ ｝〖 〗「 」『 』《 》〈 〉《》〔 〕【 】（ ）﹙﹚｛ ｝‘ ’“ ”‘ ’“ ”〞 〝— -—— ……~·•※☆★●○■□▲△▼▽⊙⊕⊖⊘⊚⊛⊜⊝◆◇◊⊿◣◢◥◤@#$%^&*+=_|\\/:;"):
    """
    return list like: [
        { "language": "punctuation", "text": },
        { "language": "not_punctuation", "text": },
    ]
    it should be a mixed result list, the order of punctuation and not_punctuation should follow orginal text
    """
    result_list = []
    index = 0
    temp_string = ""
    last_punctuation_flag =True
    if len(input_text) > 0:
        if input_text[-1] in special_punctuations:
            last_punctuation_flag = True
        else:
            last_punctuation_flag = False
    is_punctuation = True
    while True:
        current_char = input_text[index]

        if current_char in special_punctuations:
            is_punctuation = True
        else:
            is_punctuation = False

        if last_punctuation_flag != is_punctuation:
            if last_punctuation_flag == True:
                result_list.append({
                    "language": "punctuation",
                    "text": temp_string
                })
            else:
                result_list.append({
                    "language": "not_punctuation",
                    "text": temp_string
                })
            temp_string = ""

        last_punctuation_flag = is_punctuation
        temp_string += current_char

        index += 1
        if index >= len(input_text):
            break

    if len(result_list) > 0:
        if result_list[0]["text"] == "":
            result_list = result_list[1:]
    if temp_string != "":
        is_punctuation = True
        if temp_string[-1] in special_punctuations:
            is_punctuation = True
        else:
            is_punctuation = False

        if is_punctuation == False:
            result_list.append({
                "language": "punctuation",
                "text": temp_string
            })
        else:
            result_list.append({
                "language": "language",
                "text": temp_string
            })

    return result_list

def split_string_into_english_and_not_english_list(input_text):
    """
    Split a string into a list of language segments based on Chinese and English characters.

    :param input_text: The input string to split.
    :return: A list of language segments with Chinese and English text.
    """
    """
    return list like: [
        { "language": "en", "text": },
        { "language": "not_en", "text": },
    ]
    """
    result_list = []
    index = 0
    temp_string = ""
    last_punctuation_flag = False
    if len(input_text) > 0:
        if input_text[-1].isascii():
            last_punctuation_flag = True
        else:
            last_punctuation_flag = False
    is_en = True
    while True:
        current_char = input_text[index]

        if current_char.isascii():
            is_en = True
        else:
            is_en = False

        if last_punctuation_flag != is_en:
            if last_punctuation_flag == False:
                result_list.append({
                    "language": "not_en",
                    "text": temp_string
                })
            else:
                result_list.append({
                    "language": "en",
                    "text": temp_string
                })
            temp_string = ""

        last_punctuation_flag = is_en
        temp_string += current_char

        index += 1
        if index >= len(input_text):
            break

    if len(result_list) > 0:
        if result_list[0]["text"] == "":
            result_list = result_list[1:]
    if temp_string != "":
        if temp_string[-1].isascii():
            is_en = True
        else:
            is_en = False

        if is_en == False:
            result_list.append({
                "language": "not_en",
                "text": temp_string
            })
        else:
            result_list.append({
                "language": "en",
                "text": temp_string
            })

    return result_list

def string_split_by_using_yingshaoxo_method(input_text):
    """
    Split a string into language segments based on punctuations, English and not_English text.

    return list like: [
        { "language": "en", "text": },
        { "language": "not_en", "text": },
        { "language": "punctuation", "text": },
    ]
    """
    final_list = []
    punctuation_list = split_string_into_list_by_punctuations(input_text)
    for one in punctuation_list:
        if one["language"] == "punctuation":
            final_list.append({
                "language": "punctuation",
                "text": one["text"]
            })
        else:
            language_list = split_string_into_english_and_not_english_list(one["text"])
            final_list += language_list
    return final_list

def string_split_to_pure_segment_list_by_using_yingshaoxo_method(input_text):
    """
    Split a string into language segments based on punctuations, English and not_English text.

    return list like: ["how", "are", "you", "?"]
    """
    final_list = []
    a_list = string_split_by_using_yingshaoxo_method(input_text)
    for one in a_list:
        if one["language"] == "not_en":
            final_list += list(one["text"])
        else:
            final_list += [one["text"]]
    return final_list

def get_global_string_word_based_corrector_dict_by_using_yingshaoxo_method(source_text_data: str, levels: int = 10):
    global_string_dict = {}

    seperator = "☺"

    def get_x_level_dict(source_text: str, x: int):
        level_dict = {}
        tokens = string_split_to_pure_segment_list_by_using_yingshaoxo_method(source_text)
        for index in range(len(tokens)):
            if index < x:
                continue
            if index == len(tokens) - x:
                break
            current_words = ''.join(tokens[index-x: index]) + seperator + ''.join(tokens[index+1: index+x+1])
            center_word = tokens[index]
            if current_words in level_dict:
                if center_word in level_dict[current_words]:
                    level_dict[current_words][center_word] += 1
                else:
                    level_dict[current_words][center_word] = 1
            else:
                level_dict[current_words] = {center_word: 1}

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

def correct_sentence_based_on_word_by_using_yingshaoxo_method(input_text: str, levels: int = 10, source_text_data: str|None = None, global_string_corrector_dict: dict|None = None) -> any:
    if source_text_data == None:
        source_text_data = ""

    if global_string_corrector_dict != None:
        pass
    else:
        global_string_corrector_dict = {}

    input_text = "\n" * len(global_string_corrector_dict) + input_text + "\n" * len(global_string_corrector_dict)

    seperator = "☺"
    new_text = ""
    for level in global_string_corrector_dict.keys():
        tokens = string_split_to_pure_segment_list_by_using_yingshaoxo_method(input_text)
        for index in range(len(tokens)):
            if index < level or index >= len(tokens) - level:
                new_text += tokens[index]
                continue
            current_words = ''.join(tokens[index - level: index]) + seperator + ''.join(tokens[index + 1 : index + 1 + level])
            #print(current_words)
            #print(tokens[index-1], tokens[index], tokens[index+1])
            #print("____")
            if current_words in global_string_corrector_dict[level].keys():
                new_text += global_string_corrector_dict[level][current_words]
            else:
                new_text += tokens[index]
        break
    return new_text

text_generator =  ml.Yingshaoxo_Text_Generator()
text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files")
global_string_corrector_dict = get_global_string_word_based_corrector_dict_by_using_yingshaoxo_method(text, 4)
while True:
    input_text = input("What you want to say? ")
    result = correct_sentence_based_on_word_by_using_yingshaoxo_method(input_text, global_string_corrector_dict=global_string_corrector_dict)
    print(result)
