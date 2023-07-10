from urllib import response
from auto_everything.terminal import Terminal
from auto_everything.disk import Disk
from auto_everything.io import IO
disk = Disk()
io_ = IO()
terminal = Terminal()


output_model_folder = disk.join_paths(disk.get_directory_path(__file__), "out/chargpt")
input_file_folder = disk.join_paths(disk.get_directory_path(__file__), "input_txt_files")

output_txt_file = disk.join_paths(input_file_folder, "dataset.txt")
if not disk.exists(output_txt_file):
    io_.write(output_txt_file, "")

the_general_seperator = "\n\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n\n"


def encode_input(text: str):
    return io_.string_to_hex(text)

def read_database_txt_file():
    files = disk.get_files(input_file_folder, recursive=True, type_limiter=[".txt"])
    content = ""
    for file in files:
        content += open(file, 'r').read() # don't worry we won't run out of file handles
    return encode_input(content)

def decode_response(context_text: str, text: str):
    text = io_.hex_to_string(text)
    text = text[len(context_text):]
    #response = text.replace("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n", "\n\n")
    response = text.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n")[0].strip()

    final_response = terminal.run_python_code(code=response)
    if final_response.strip() == "":
        final_response = response

    return final_response


def general_text_wrapper(who_said: str, language: str, content: str):
    text = ""
    text += content
    text += the_general_seperator
    return text

def handle_yingshaoxo_ai_text(text: str):
    text = text.strip()
    if (text == ""):
        return
    with open(output_txt_file, "a", encoding="utf-8", errors="ignore") as f:
        text_template = f"""
print('''
{text}
'''.strip())
        """.strip()
        f.write(general_text_wrapper(who_said="yingshaoxo", language="python", content=text_template))

def handle_pi_ai_text(text: str):
    text = text.strip()
    if (text == ""):
        return
    with open(output_txt_file, "a", encoding="utf-8", errors="ignore") as f:
        f.write(general_text_wrapper(who_said="yingshaoxo", language="python", content=text))
