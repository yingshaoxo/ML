import random
import common_functions

import transformers
from auto_everything.disk import Disk
from auto_everything.ml import Yingshaoxo_Text_Generator
from auto_everything.python import Python
disk = Disk()
python = Python()
yingshaoxo_text_generator = Yingshaoxo_Text_Generator(common_functions.input_file_folder)


model_saving_folder_path = disk.join_paths(disk.get_directory_path(__file__), "bert_model_saving_place")


model_name = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"
the_main_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
#the_main_tokenizer = transformers.PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

try:
    the_main_model = transformers.PegasusForConditionalGeneration.from_pretrained(model_saving_folder_path).to("cuda")
    print("\n\nusing local model\n\n")
except Exception as e:
    print(e)
    the_main_model = transformers.PegasusForConditionalGeneration.from_pretrained(model_name).to("cuda")
    the_main_model.save_pretrained(model_saving_folder_path)
    print("\n\nusing remote model\n\n")


traning_step = 0
def train_for_once(source_text: str, target_text: str):
    global traning_step

    input_ids = the_main_tokenizer(
        source_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    target = the_main_tokenizer(target_text, return_tensors="pt").input_ids

    loss = the_main_model(input_ids=input_ids.to("cuda"), decoder_input_ids=target.to("cuda"), labels=target.to("cuda")).loss
    loss.backward()

    traning_step += 1
    if traning_step % 100 == 0:
        print(f"Step {traning_step}: Loss {loss}")

def save_model():
    print("Saving model...")
    the_main_model.save_pretrained(model_saving_folder_path)
    print("Model saved.")


def train():
    the_main_model.learning_rate = 0.00001

    txt_source = common_functions.read_database_txt_file()
    text_lines = [one.strip() for one in txt_source.split("\n") if one.strip() != ""]
    while True:
        input_line = ""
        target_line = ""
        for i in range(500):
            target_line = random.choice(text_lines)
            input_line = yingshaoxo_text_generator.get_random_text_deriation_from_source_text(source_text=target_line, random_remove_some_characters=False, random_add_some_characters=False, random_char_source_text=txt_source)

            train_for_once(
                input_line,
                target_line
            )

        print(input_line)
        print(target_line)

        save_model()

def predict_once(input_text: str) -> str:
    input_ids = the_main_tokenizer(
        input_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    outputs = the_main_model.generate(input_ids.to("cuda"))
    return the_main_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def test():
    text = "hi"
    while True:
        if text.strip() != "":
            result = predict_once(text)
            print()
            print(result)
            print()

        text = input("What you want to say?\n")


class Run():
    def train(self):
        train()
    
    def test(self):
        test()


python.fire(Run)