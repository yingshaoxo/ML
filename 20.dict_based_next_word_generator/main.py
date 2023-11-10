from auto_everything.disk import Disk
from auto_everything.io import IO
disk = Disk()
io_ = IO()

source_text = ""

files = disk.get_files("../18.fake_ai_asistant/input_txt_files", type_limiter=[".txt"])
for file in files:
    source_text += io_.read(file)

"""
# dict based next word generator

```
One word Predict next word
One word predict next word
Two words predict next word
Three words predict next word
... words predict next word
```

When you use it, use it from bottom to top, use longest sequence to predict the next word first.
> This method was created by yingshaoxo
"""
global_string_dict = {}

def get_x_level_dict(source_text: str, x: int):
    level_dict = {}
    for index, _ in enumerate(source_text):
        if index < (x-1):
            continue
        if index == len(source_text) - x:
            break
        current_chars = source_text[index-(x-1): index+1]
        next_char = source_text[index+1]
        if current_chars in level_dict:
            if next_char in level_dict[current_chars]:
                level_dict[current_chars][next_char] += 1
            else:
                level_dict[current_chars][next_char] = 1
        else:
            level_dict[current_chars] = {next_char: 1}

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

global_string_dict = {
    #2: pure_level2_dict,
    #1: pure_level1_dict,
}
max_level = 20
for level in reversed(list(range(1, max_level))):
    global_string_dict[level] = get_x_level_dict(source_text, level)


def predict_next_char(input_text: str):
    for level in global_string_dict.keys():
        last_chars = input_text[len(input_text)-level:]
        if last_chars in global_string_dict[level].keys():
            return global_string_dict[level][last_chars]
    return None


def predict_next_x_chars(input_text: str, x: int):
    complete_text = input_text
    for _ in range(x):
        result = predict_next_char(complete_text)
        if result == None:
            break
        else:
            complete_text += result
    return complete_text


while True:
    input_text = input("What you want to say? ")
    result = predict_next_x_chars(input_text, 30)
    print(result)
