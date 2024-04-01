import json
import torch
import numpy as np
from tqdm import tqdm
from llasm.model.encoder import *
import wandb
import argparse
import os
device = torch.device("cuda")

def normalized_l2_norm(vector, epsilon=1e-5):
    norm = np.linalg.norm(vector)
    normalized_norm = norm / np.sqrt(len(vector)) + epsilon
    return normalized_norm

def calculate_l2_norms(list_of_vectors):
    norms = [normalized_l2_norm(v) for v in list_of_vectors]
    return norms

def normalize_norms(norms):
    min_norm = min(norms)
    max_norm = max(norms)
    normalized = [(norm - min_norm) / (max_norm - min_norm) for norm in norms]
    return normalized

def mean_norms(rank):
    norms = []
    for data in rank:
        norms.append(np.mean(data))
    return norms

class NamingSupervisionDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, tokenizer_path):
        self.norms = []
        self.asms = []
        self.sites = []
        self.tokenizer = AsmTokenizer.from_pretrained(tokenizer_path)
        file_list = os.listdir(data_path)
        for file_path in file_list:
            for line in open(os.path.join(data_path, file_path)).readlines():
                data = json.loads(line)
                self.norms.append(data["value"])
                self.asms.append(data["asm"])
                self.sites.append(data["site"])

    def tokenizer(self, data_batch):
        norms = []
        asms = []
        sites = []
        
        for data in data_batch:
            norms.append(data[0])
            asms.append(self.tokenizer.encode_function(data[1]))
            sites.append(self.tokenizer.encode_function(data[2], 768))

        asm_inputs = self.tokenizer.pad(asms, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        site_inputs = self.tokenizer.pad(sites, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        return norms, asm_inputs, site_inputs
        
    def __len__(self):
        return len(self.asms)
    
    def __getitem__(self, i):
        norms = self.norms[i]
        asms = self.asms[i]
        sites = self.sites[i]
        return norms, asms, sites

def load_model():

    encoder = Supervisor.from_pretrained(args.encoder).cuda()
    supervision = BertRegression(args.in_dim)
    supervision = supervision.cuda()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in supervision.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in supervision.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    return encoder, supervision, optimizer

def eval(encoder, supervision, criterion, global_steps):
    supervision.eval()
    total_loss = 0
    num_batches = len(test_dataloader)

    with torch.no_grad():
        for norms, asms, sites in tqdm(test_dataloader):
            asms = asms.to(device)
            norms_tensor = torch.tensor(norms, dtype=torch.float32).to(device)
            asm_embeddings = encoder(**asms)
            site_embeddings = encoder(**sites)
            regression = supervision(torch.cat((asm_embeddings, site_embeddings), dim=0)).squeeze()
            loss = criterion(regression, norms_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    wandb.log({'val_loss': avg_loss}, step=global_steps)
    supervision.train()

def train():
    criterion = nn.MSELoss()
    wandb.init(project=args.wandb_proj, name=args.wandb_name)
    encoder, supervision, optimizer = load_model()
    global_steps = 0
    supervision.train()
    for _ in tqdm(range(args.epochs)):
        for norms, asms, sites in tqdm(train_dataloader):
            asms = asms.to(device)
            sites = sites.to(device)
            norms_tensor = torch.tensor(norms, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                asm_embeddings = encoder(**asms)
                site_embeddings = encoder(**sites)

            regression = supervision(torch.cat((asm_embeddings, site_embeddings), dim=0)).squeeze()
            loss = criterion(regression, norms_tensor)
            loss.backward()

            optimizer.step()
            wandb.log({'loss':loss},step=global_steps)
            if global_steps % args.eval_steps == 0:
                eval(encoder, supervision, criterion, global_steps)
                torch.save(supervision.state_dict(),os.path.join(args.save_path, f"supervisor_{global_steps}" + ".pth"))
            global_steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llasm_supervision")
    parser.add_argument("--wandb_name", type=str, default='supervision',  help='the name of wandb')
    parser.add_argument("--wandb_proj", type=str, default='llasm',  help='the project of wandb')
    ########### models
    parser.add_argument("--encoder", type=str, default='../../encoder/llasm/encoder',  help='the path of asm encoder model')
    parser.add_argument("--tokenizer", type=str, default='../../encoder/llasm/tokenizer', help='the path of encode tokenizer')
    parser.add_argument("--save_path", type=str, default='../../models/supervisor', help='save model path')
    ########### datapath
    parser.add_argument("--train_path", type=str, default='./save/train',  help='the path of train dataset')
    parser.add_argument("--valid_path", type=str, default='./save/valid',  help='the path of valid dataset')
    ########### train arguments
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate')
    parser.add_argument("--weight_decay", type=int, default = 1e-4, help='regularization weight decay')
    parser.add_argument("--in_dim", type=int, default=2048, help='learning rate')
    parser.add_argument("--epochs", type=int, default=20, help='number of training epochs')
    parser.add_argument("--eval_steps", type=int, default=1000, help='number of eval steps')
    args = parser.parse_args([])

    train_data = NamingSupervisionDataset(args.train_path, args.tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=128, num_workers=4, shuffle=True, collate_fn=train_data.tokenizer)

    test_data = NamingSupervisionDataset(args.test_path, args.tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=128, num_workers=4, shuffle=True, collate_fn=test_data.tokenizer)

    train()


        

