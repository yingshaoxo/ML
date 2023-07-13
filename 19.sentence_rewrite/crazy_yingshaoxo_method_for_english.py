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


model_name = "google/roberta2roberta_L-24_discofuse"
training_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
#training_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
try:
    training_bert2bert_encoder_and_decoder_model = transformers.EncoderDecoderModel.from_pretrained(model_saving_folder_path).to("cuda")
    print("\n\nusing local model\n\n")
except Exception as e:
    print(e)
    training_bert2bert_encoder_and_decoder_model = transformers.EncoderDecoderModel.from_pretrained(model_name).to("cuda")
    training_bert2bert_encoder_and_decoder_model.save_pretrained(model_saving_folder_path)
    print("\n\nusing remote model\n\n")


traning_step = 0
def train_for_once(source_text: str, target_text: str):
    global traning_step

    input_ids = training_tokenizer(
        source_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    target = training_tokenizer(target_text, return_tensors="pt").input_ids

    loss = training_bert2bert_encoder_and_decoder_model(input_ids=input_ids.to("cuda"), decoder_input_ids=target.to("cuda"), labels=target.to("cuda")).loss
    loss.backward()

    traning_step += 1
    if traning_step % 100 == 0:
        print(f"Step {traning_step}: Loss {loss}")

def save_model():
    print("Saving model...")
    training_bert2bert_encoder_and_decoder_model.save_pretrained(model_saving_folder_path)
    print("Model saved.")


def train():
    training_bert2bert_encoder_and_decoder_model.learning_rate = 0.00001

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
    input_ids = training_tokenizer(
        input_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    outputs = training_bert2bert_encoder_and_decoder_model.generate(input_ids.to("cuda"))
    return training_tokenizer.decode(outputs[0])

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