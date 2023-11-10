from auto_everything.ml import ML, string_split_to_pure_segment_list_by_using_yingshaoxo_method
from auto_everything.disk import Disk
ml = ML()
disk = Disk()
text_generator = ml.Yingshaoxo_Text_Generator()

def get_global_string_word_based_generator_dict_by_using_yingshaoxo_method(source_text_data: str, levels: int = 10):
    global_string_dict = {}

    def get_x_level_dict(source_text: str, x: int):
        level_dict = {}
        tokens = string_split_to_pure_segment_list_by_using_yingshaoxo_method(source_text, without_punctuation=True)
        for index in range(len(tokens)):
            if index < x:
                continue
            if index == len(tokens) - x:
                break
            current_words = ' '.join(tokens[index-x: index])
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

    return global_string_dict

def generate_sentence_based_on_word_by_using_yingshaoxo_method(input_text: str, x: int=64, levels: int = 10, source_text_data: str|None = None, global_string_word_based_generator_dict: dict|None = None) -> any:
    if source_text_data == None:
        source_text_data = ""

    if global_string_word_based_generator_dict != None:
        pass
    else:
        global_string_word_based_generator_dict = {}

    #input_text = "\n" * len(global_string_word_based_generator_dict) + input_text

    original_words = string_split_to_pure_segment_list_by_using_yingshaoxo_method(input_text, without_punctuation=True)
    words = original_words.copy()
    for _ in range(x):
        for level in global_string_word_based_generator_dict.keys():
            current_words = " ".join(words[-level:])
            if current_words in global_string_word_based_generator_dict[level].keys():
                words += string_split_to_pure_segment_list_by_using_yingshaoxo_method(global_string_word_based_generator_dict[level][current_words])
                break
    return " ".join(words[len(original_words):])

text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files")
text = text.replace("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n", "\n\n\n")
global_string_word_based_generator_dict = get_global_string_word_based_generator_dict_by_using_yingshaoxo_method(text, 8)
while True:
    input_text = input("What you want to say? \n")
    result = generate_sentence_based_on_word_by_using_yingshaoxo_method(input_text, x=32, global_string_word_based_generator_dict=global_string_word_based_generator_dict)
    print(result.strip(), end="")
