"""
Trains a character-level language model.
"""
"""
pip install -e ./min_GPT 
"""

import os
import sys

import torch
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from auto_everything.disk import Disk
disk = Disk()

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
    C.model.model_type = 'gopher-44m'#'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

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
    model = GPT(config.model)

    # construct the trainer object
    config.trainer.device = "cuda" #"cpu"
    trainer = Trainer(config.trainer, model, train_dataset)

    # load model if possible
    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    if (os.path.exists(ckpt_path)):
        ck = torch.load(ckpt_path)
        model.load_state_dict(ck) 
    model.transformer.to(config.trainer.device)
    model.lm_head.to(config.trainer.device)

    print("\n\n")
    all_input_text = ""

    # iteration callback
    def batch_end_callback(trainer):
        global all_input_text

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 200 == 0:
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)

            input_text = input("\nWhat you want to say? \n")
            all_input_text += input_text + common_functions.the_general_seperator

            common_functions.handle_yingshaoxo_ai_text(text=input_text)

            model.eval()
            with torch.no_grad():
                # sample from the model...
                input_context = common_functions.encode_input(all_input_text[-8000:])
                x = torch.tensor([train_dataset.stoi[s] for s in input_context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                response = common_functions.decode_response(context_text=all_input_text[-8000:], text=completion)
                print()
                print(response)
                print("\n\n---------\n\n")

            common_functions.handle_pi_ai_text(text=response)

            input("Done the editing to the response in txt file? (hit enter to go on)")

            # reload txt data
            text = common_functions.read_database_txt_file()
            trainer.train_dataset = CharDataset(config.data, text)

            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()