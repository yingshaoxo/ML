import common_functions

from auto_everything.disk import Disk
from auto_everything.io import IO
from auto_everything.language import Language
from auto_everything.ml import Yingshaoxo_Text_Generator
from auto_everything.python import Python
io_ = IO()
disk = Disk()
language = Language()
python = Python()
yingshaoxo_text_generator = Yingshaoxo_Text_Generator(common_functions.input_file_folder, use_machine_learning=True)

def replace_it_with_new_txt(input_text: str) -> str:
    found = yingshaoxo_text_generator.search_and_get_following_text_in_a_exact_way(input_text=input_text, also_want_the_current_line=False, extremly_accrate_mode=True)
    if found.strip() == "":
        return found.strip()
    else:
        splits = language.seperate_text_to_segments(found)
        if len(splits) > 1:
            return splits[0]["text"] + splits[1]["text"]
        else:
            return splits[0]["text"]


"""
txt_source = io_.read("/home/yingshaoxo/Downloads/Super Student.txt")
text_lines = [one.strip() for one in txt_source.split("\n") if one.strip() != ""]

final_text_list = []
for one in text_lines[120:]:
    splits = language.seperate_text_to_segments(one)
    for part in splits:
        old_text = part["text"]
        if part["is_punctuation_or_space"] == False:
            new_text = replace_it_with_new_txt(old_text)
            if new_text != "":
                similarity = yingshaoxo_text_generator.get_similarity_of_two_sentences(old_text, new_text, use_both_machine_learning_and_traditional_method=True)
                if (similarity > 0.7):
                    print(f"old_text: {old_text}")
                    print(f"new_text: {new_text}")
                    print(similarity)

                    input("Hit enter to go on...")
            else:
                final_text_list.append(old_text)

final_text = "\n".join(final_text_list)
"""


text = "hi"
while True:
    if text.strip() != "":
        result = replace_it_with_new_txt(text)
        print()
        print(result)
        print()

    text = input("What you want to say?\n")