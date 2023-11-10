from time import sleep

from auto_everything.ml import Yingshaoxo_Text_Generator
from auto_everything.terminal import Terminal
terminal = Terminal()

yingshaoxo_text_generator = Yingshaoxo_Text_Generator(
    input_txt_folder_path="/home/yingshaoxo/CS/ML/18.fake_ai_asistant/input_txt_files",
    use_machine_learning=False
)

def decode_response(text: str, chat_context: str):
    #print("`"+text+"`")
    splits = text.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n")
    if (len(splits) > 1):
        response = splits[1].strip()
    elif (len(splits) == 1):
        response = splits[0].strip()
    else:
        response = ""
    new_code = f"""
chat_context = '''{chat_context}'''

{response}
"""
    final_response = terminal.run_python_code(code=new_code)
    if final_response.strip() == "":
        final_response = response
    final_response = "\n".join([one for one in final_response.split("\n") if not one.strip().startswith("__**")])
    return final_response

sleep(7)
print("\n\n")
all_input_text = ""
while True:
    input_text = input("What you want to say? \n")

    all_input_text += input_text + "\n"
    real_input = all_input_text[-800:].strip()
    response = yingshaoxo_text_generator.search_and_get_following_text_in_a_exact_way(input_text=real_input, quick_mode=True)
    response = decode_response(text=response, chat_context=all_input_text)
    all_input_text += response + "\n"

    print("\n\n---------\n\n")
    print(response)
    print("\n\n---------\n\n")
