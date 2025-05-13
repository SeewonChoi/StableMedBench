from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

import random
from time import time
from tqdm import tqdm

import wandb
from argparse import ArgumentParser

from dataset import sepsis_loader
from models import CustomGPT, CustomMamba

EPS = 1e-2

class SickDataset(Dataset):
    def __init__(self, data):
        self.samples = data[['eventval', 'time']]
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        label = torch.tensor(1)
        event = list(sample['eventval'])
        time = list(sample['time'])
        return event, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        labels = torch.stack(labels, dim=0)
        return events, times, labels

class Trainer():
    def __init__(self, model, train_loader, test_loader, val_loader, learning_rate, gpu, save_model=True):
        if gpu >= 0 and torch.cuda.is_available(): device = torch.device("cuda:%d" % gpu)
        else: device = torch.device("cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.save_model = save_model
        self.best_acc = 0
        self.criterion = F.cross_entropy
        self.ignore_seq = 4

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/decomp_tokenizer.json")
        self.tokenizer.eos_token = '[EOS]'
        self.tokenizer.sep_token = '[SEP]'
        self.tokenizer.bos_token = '[BOS]'
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.cls_token = '[CLS]'
        self.tokenizer.mask_token = '[MASK]'

        self.network = model(vocab_size = len(self.tokenizer)).to(self.device)
        self.optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.network.parameters())), lr=learning_rate)

    def get_inputs(self, inputs, times):
        max_len = max(len(i) for i in inputs)
        tokens = self.tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
        sequences = tokens["input_ids"].to(self.device)   
        attention_masks = tokens["attention_mask"].to(self.device) 

        times = [F.pad(torch.tensor(r, dtype=torch.float32), (0, max_len - len(r))).round(decimals=2)  for r in times]
        times = torch.stack(times, dim=0).to(self.device)  

        return sequences, times, attention_masks

    def train_epoch(self, epoch):
        train_loss = 0
        total_correct = 0
        num_items = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        self.network.train()
        for (i, (events, times, _)) in enumerate(iter):
            self.optimizer.zero_grad()

            input_ids, times, attention_mask = self.get_inputs(events, times)
            target = input_ids[:, 1+self.ignore_seq:]

            output = self.network(input_ids[:, :-1], times[:, :-1], attention_mask[:, :-1])
            output = output[:, self.ignore_seq:].squeeze(-1)
            
            loss = self.criterion(output.flatten(0, 1), target.flatten())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            avg_loss = train_loss / (i + 1)

            total_correct += (output.argmax(dim=-1) == target).float().sum()
            num_items += target.numel()
            train_acc = (total_correct / num_items)

            iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {total_correct}/{num_items} {train_acc:.4f}")
        return avg_loss

    def test_epoch(self, epoch):
        total_correct = 0
        num_items = 0
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        self.network.eval()
        for (i, (events, times, _)) in enumerate(iter):
            with torch.no_grad():
                input_ids, times, attention_mask = self.get_inputs(events, times)
                target = input_ids[:, 1+self.ignore_seq:]

                output = self.network(input_ids[:, :-1], times[:, :-1], attention_mask[:, :-1])
                output = output[:, self.ignore_seq:].squeeze(-1)

                total_correct += (output.argmax(dim=-1) == target).float().sum()
                num_items += target.numel()
                test_acc = (total_correct / num_items)
                iter.set_description(f"[Test Epoch {epoch}] Acc: {total_correct}/{num_items} {test_acc:.4f}")
        return test_acc

    def train(self, n_epochs):
        for epoch in range(1, n_epochs+1):
            t0 = time()
            train_loss = self.train_epoch(epoch)
            t1 = time()
            test_acc = self.test_epoch(epoch)
            t2 = time()

            wandb.log({
                "train_loss": train_loss,
                "test_acc": test_acc,
                "train time": t1 - t0,
                "test time": t2 - t1,
                "epoch time": t2 - t0})
        
            if self.save_model and test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.network.state_dict(), "checkpoints/decomp_eventval_mamba.pth")


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("sepsis")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model", type=str, choices=["gpt", "mamba"], default="mamba")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # Setup parameters
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model == "gpt":
        model = CustomGPT
    elif args.model == "mamba":
        model = CustomMamba

    # Load data
    (train_loader, val_loader, test_loader) = sepsis_loader(args.batch_size, SickDataset, seed=args.seed)
    print("LOADED")
    trainer = Trainer(model, train_loader, test_loader, val_loader, args.learning_rate, args.gpu)

    # setup wandb
    config = vars(args)
    run = wandb.init(
        project=f"MCMED_decomp_pretrain",
        config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)

    run.finish()