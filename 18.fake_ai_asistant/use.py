"""
Trains a character-level language model.
"""

import os
import sys
from typing import final

import torch
from torch.utils.data import Dataset

from auto_everything.terminal import Terminal
terminal = Terminal()

from mingpt.model import GPT
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

import common_functions

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = common_functions.output_model_folder

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = common_functions.model_type

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = common_functions.read_database_txt_file()
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    device = common_functions.computing_device #"cpu"
    model = GPT(config.model)

    # load model if possible
    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    if (os.path.exists(ckpt_path)):
        ck = torch.load(ckpt_path)
        model.load_state_dict(ck) 
    model.transformer.to(device)
    model.lm_head.to(device)

    # iteration callback
    def batch_end_callback(context_text: str):
        response = ""

        # evaluate both the train and test score
        model.eval()
        with torch.no_grad():
            # sample from the model...
            context = common_functions.encode_input(context_text)
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(device)
            y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
            completion = ''.join([train_dataset.itos[int(i)] for i in y])
            response += completion

        return common_functions.decode_response(context_text=context_text, text=response)

    
    print("\n\n")
    all_input_text = ""
    while True:
        input_text = input("What you want to say? \n")
        all_input_text += input_text + common_functions.the_general_seperator
        response = batch_end_callback(context_text=all_input_text[-8000:])
        print("\n\n---------\n\n")
        print(response)
        print("\n\n---------\n\n")
