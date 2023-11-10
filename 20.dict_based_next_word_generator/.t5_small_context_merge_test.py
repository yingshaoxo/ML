from auto_everything.ml import ML, split_string_into_list_by_punctuations
from auto_everything.disk import Disk
from auto_everything.terminal import Terminal
ml = ML()
disk = Disk()
terminal = Terminal()

text_generator =  ml.Yingshaoxo_Text_Generator()

text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files")
text_list = text.split("\n\n")
text_list = [one.strip() for one in text_list]
data = []
for one in text_list:
    if one.strip() == "":
        continue
    target = one
    sentence_segment_list = split_string_into_list_by_punctuations(target, not_include_punctuations="\n' _")
    new_sentence_segment_list = []
    for one in sentence_segment_list:
        if one["language"] != "punctuation":
            new_sentence_segment_list.append(one["text"])
    source = " ".join(new_sentence_segment_list)
    data.append({
        "input": source,
        "output": target,
    })
    #print("**********")
    #print(target)
    #print("_______")
    #print(source)
    #print("**********")


import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import os
os.environ["WANDB_DISABLED"] = "true"

tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-tiny")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-tiny")
#model = AutoModelForSeq2SeqLM.from_pretrained("./checkpoints/t5-efficient-tiny-text-formater.pt")


# Papare data
data = [{"input": "Hi you ", "output": "Hi,you!"}]
#dataset = Dataset.from_json("dataset.json")
def prepare_training_data(example):
    input_ids = tokenizer(example["input"], return_tensors="pt", padding="max_length", truncation=True, max_length=256).input_ids
    output_ids = tokenizer(example["output"], return_tensors="pt", padding="max_length", truncation=True, max_length=256).input_ids

    return {
        "input_ids": input_ids,
        "decoder_input_ids": output_ids,
        "labels": output_ids,
    }
dataset = Dataset.from_list(data).map(prepare_training_data, batched=True, )


# Do the traning
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        learning_rate= 0.00005,
        num_train_epochs=10,
        output_dir="./checkpoints",
    ),
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained("./checkpoints/t5-efficient-tiny-text-formater.pt")


# Make prediction
model = AutoModelForSeq2SeqLM.from_pretrained("./checkpoints/t5-efficient-tiny-text-formater.pt")
while True:
    prompt = input("What you want to say? ")
    outputs = model.generate(input_ids=tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=256).input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)



"""
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("mrm8488/tiny-distilbert-t5-v1")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/tiny-distilbert-t5-v1")

# Define input text
text = "translate English to French: Hello, how are you?"

# Tokenize the input text
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate the output
outputs = model.generate(input_ids)

# Decode the output tokens
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated text
print(output_text)
"""
