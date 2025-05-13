from transformers import PreTrainedTokenizerFast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAUPRC

import random
from time import time
from tqdm import tqdm
import wandb
from argparse import ArgumentParser
from collections import OrderedDict

from models_linear import CustomGPT, CustomMamba
from mcmed_loader import decomp_loader, sepsis_loader
from ehrshot_loader import hyperkalemia_loader, hypoglycemia_loader

class LabDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'start', 'Label', 'age_str', 'gender_str', 'race_str', 'ethnicity_str']]
        for _, d in data.iterrows():
            if d['Label'] == 1: 
                self.samples.append(d)
                labels.append(d['Label'])
            else: 
                if random.random() < 1.0: 
                    self.samples.append(d)
                    labels.append(d['Label'])
        
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)

        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        print(f"Original: {counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[self.index_map[idx]] 
        label = float(sample['Label'])
        event = list(sample['eventval'])
        time = list(sample['start'])
        demo = [sample['age_str'], sample['gender_str'], sample['race_str'], sample['ethnicity_str']]
        return event, demo, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        demos = [item[1] for item in batch]
        times = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        labels = torch.tensor(labels)
        return events, demos, times, labels

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
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3323./196).to(self.device))

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/eventval_tokenizer.json")
        self.tokenizer.eos_token = '[EOS]'
        self.tokenizer.sep_token = '[SEP]'
        self.tokenizer.bos_token = '[BOS]'
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.cls_token = '[CLS]'
        self.tokenizer.mask_token = '[MASK]'

        self.N = 4

        self.network = model(vocab_size = len(self.tokenizer), full=False).to(self.device)
        if args.model == 'load':
            lmhead_state_dict = torch.load(f"checkpoints/gpt_eventval.pth", map_location=self.device, weights_only=True)
            new_state_dict = OrderedDict()
            for key, value in lmhead_state_dict.items():
                if ('wte' not in key) and (('lm_head' not in key)):
                    new_key = key.replace('model.transformer', 'model')
                    new_state_dict[new_key] = value
            self.network.load_state_dict(new_state_dict, strict=False)

        self.classifier = nn.Linear(len(self.tokenizer)*self.N, 1).to(self.device)

        params = list(filter(lambda p: p.requires_grad, self.network.parameters())) + list(self.classifier.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)


    def train_epoch(self, epoch):
        train_loss = 0
        total_correct = 0
        num_items = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        self.network.train()
        for (i, (events, demos, times, target)) in enumerate(iter):
            self.optimizer.zero_grad()
            target = target.to(self.device)

            next_probs = self.demo_inference(events, demos, times)
            output = self.classifier(next_probs.flatten(1)).squeeze(-1)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            avg_loss = train_loss / (i + 1)
            
            output = F.sigmoid(output)
            total_correct += (output.round()==target).float().sum() 
            num_items += output.shape[0]
            train_acc = (total_correct / num_items)
            iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {total_correct}/{num_items} {train_acc:.4f}")
        return train_loss / len(self.train_loader)

    def test_epoch(self, epoch):
        total_correct = 0
        num_items = 0
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        self.network.eval()
        
        auroc = BinaryAUROC().to(self.device)
        f1 = BinaryF1Score().to(self.device)
        precision = BinaryPrecision().to(self.device)
        recall = BinaryRecall().to(self.device)
        confusion_matrix = BinaryConfusionMatrix().to(self.device)
        auprc = BinaryAUPRC().to(self.device)
        
        for (i, (events, demos, times, target)) in enumerate(iter):
            with torch.no_grad():
                target = target.to(self.device)

                next_tokens = self.demo_inference(events, demos, times)
                output = self.classifier(next_tokens.flatten(1)).squeeze(-1)

                output = F.sigmoid(output)
                preds = output.round()

                total_correct += (preds==target).float().sum() 
                num_items += output.shape[0]
                test_acc = (total_correct / num_items)

                target = target.long()
                f1.update(preds, target)
                precision.update(preds, target)
                recall.update(preds, target)
                auroc.update(output, target)
                auprc.update(output, target)
                confusion_matrix.update(preds, target)

                iter.set_description(f"[Test Epoch {epoch}] Acc: {total_correct}/{num_items} {test_acc:.4f}")
        
        f1 = f1.compute()
        auc = auroc.compute()
        precision = precision.compute()
        recall = recall.compute()
        conf_matrix = confusion_matrix.compute()
        auprc = auprc.compute()

        print("Confusion Matrix:\n", conf_matrix)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")

        return test_acc, auc, f1, precision, recall, auprc

    def inference(self, inputs, demos, times, temperature=1.0):
        next_probabilities = []

        for i in range(self.N):
            max_len = max(len(i) for i in inputs)
            tokens = self.tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
            sequences = tokens["input_ids"].to(self.device)   
            attention_masks = tokens["attention_mask"].to(self.device) 

            seq_times = [F.pad(torch.tensor(r, dtype=torch.float32), (0, max_len - len(r))).round(decimals=2)  for r in times]
            seq_times = torch.stack(seq_times, dim=0).to(self.device) 

            next_ids = self.network(sequences, seq_times, attention_masks)
            
            next_token_logits = next_ids[:, -1, :] / temperature
            if temperature > 0:
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_logits, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            inputs = [inputs_i[-1023:] + [next_i] for (inputs_i, next_i) in zip(inputs, self.tokenizer.convert_ids_to_tokens(next_token))]
            times = [t_i[-1023:] + [t_i[-1] + 0.25] for t_i in times]
            next_probabilities.append(F.softmax(next_ids[:, -1, :], dim=-1))
        next_probabilities = torch.stack(next_probabilities, dim=1)
        return next_probabilities

    def demo_inference(self, inputs, demos, times, temperature=1.0):
        next_probabilities = []

        for i in range(self.N):
            max_len = max(len(i) for i in inputs)
            demo_inputs = [demo + inputs_i for demo, inputs_i in zip(demos, inputs)]

            tokens = self.tokenizer(demo_inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
            sequences = tokens["input_ids"].to(self.device)   
            attention_masks = tokens["attention_mask"].to(self.device) 

            seq_times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len - len(r))).round(decimals=2)  for r in times]
            seq_times = torch.stack(seq_times, dim=0).to(self.device) 

            next_ids = self.network(sequences, seq_times, attention_masks)
            
            next_token_logits = next_ids[:, -1, :] / temperature
            if temperature > 0:
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_logits, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            inputs = [inputs_i[-1019:] + [next_i] for (inputs_i, next_i) in zip(inputs, self.tokenizer.convert_ids_to_tokens(next_token))]
            times = [t_i[-1019:] + [t_i[-1] + 0.25] for t_i in times]
            next_probabilities.append(next_ids[:, -1, :])
        next_probabilities = torch.stack(next_probabilities, dim=1)
        return next_probabilities

    def train(self, n_epochs):
        for epoch in range(1, n_epochs+1):
            t0 = time()
            train_loss = self.train_epoch(epoch)
            t1 = time()
            test_acc, auc, f1, precision, recall, auprc = self.test_epoch(epoch)
            t2 = time()

            wandb.log({
                "train_loss": train_loss,
                "test_acc": test_acc,
                "train time": t1 - t0,
                "test time": t2 - t1,
                "epoch time": t2 - t0,
                "auc": auc,
                "auprc": auprc,
                "f1": f1,
                "precision": precision,
                "recall": recall,})

            if self.save_model and test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.network.state_dict(), f"checkpoints/hg_linear_{args.model}_{args.seed}.pth")

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("sepsis")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", type=str, choices=["gpt", "mamba"], default="gpt")
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
    (train_loader, val_loader, test_loader) = hypoglycemia_loader(args.batch_size, LabDataset, seed=args.seed)
    trainer = Trainer(model, train_loader, test_loader, val_loader, args.learning_rate, args.gpu)

    # setup wandb
    config = vars(args)
    config['task'] = 'hg'
    run = wandb.init(
        project=f"EHRSHOT_LINEAR",
        name=f'hg_{args.model}_{args.seed}',
        config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)

    run.finish()