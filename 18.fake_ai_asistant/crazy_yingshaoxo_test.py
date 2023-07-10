from auto_everything.ml import Yingshaoxo_Text_Generator
from auto_everything.terminal import Terminal
terminal = Terminal()

import common_functions

#common_functions.format_data_set()
#exit()

yingshaoxo_text_generator = Yingshaoxo_Text_Generator(
    input_txt_folder_path="/home/yingshaoxo/CS/ML/18.fake_ai_asistant/input_txt_files"
)

def decode_response(text: str):
    response = text.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n")[0].strip()
    final_response = terminal.run_python_code(code=response)
    if final_response.strip() == "":
        final_response = response
    return final_response

print("\n\n")
all_input_text = ""
while True:
    input_text = input("What you want to say? \n")
    all_input_text += input_text + common_functions.the_general_seperator
    real_input = all_input_text[-8000:].strip()
    response = yingshaoxo_text_generator.search_and_get_following_text(input_text=real_input)
    response = decode_response(text=response)
    print("\n\n---------\n\n")
    print(response)
    print("\n\n---------\n\n")