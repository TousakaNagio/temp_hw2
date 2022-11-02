import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from datasets import MCDataset
from model import MCModel
from utils import same_seeds, save_model, get_config



def train(model, accelerator, data_loader, optimizer, args, scheduler=None):
    model.train()
    train_loss, train_acc = [], []

    for idx, batch in enumerate(tqdm(data_loader)):
        ids, input_ids, token_type_ids, attention_masks, labels = batch
        loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=labels)
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        loss = loss / args.accumulation_steps
        accelerator.backward(loss)

        if ((idx + 1) % args.accumulation_steps == 0) or (idx == len(data_loader) - 1):
            # accelerate 1 time every accumulation_steps
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            #if scheduler is not None:
            #    scheduler.step()

        train_loss.append(loss.item())
        train_acc.append(acc)

    # train_loss = sum(train_loss) / len(train_loss)
    # train_acc = sum(train_accs) / len(train_accs)

    return sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc)


@torch.no_grad()
def validate(model, data_loader):
    model.eval()
    valid_loss, valid_acc = [], []

    for batch in tqdm(data_loader):
        ids, input_ids, token_type_ids, attention_masks, labels = batch
        loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=labels)
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        valid_loss.append(loss.item())
        valid_acc.append(acc)
    # valid_loss = sum(valid_loss) / len(valid_loss)
    # valid_acc = sum(valid_acc) / len(valid_acc)

    return sum(valid_loss) / len(valid_loss), sum(valid_acc) / len(valid_acc)


def main(args):
    same_seeds(args.seed)
    
    # Config
    if args.from_scratch:
        config = get_config()
    else:
        config = AutoConfig.from_pretrained(args.model_name, return_dict=False)
    print(config)

    # Tokenizer
    # tokenizer = AutoTokenizer.save_pretrained('pretrained_token')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=config, model_max_length=args.max_len, use_fast=True)
    # tokenizer.save_pretrained('./pretrained_token')

    # Prepare dataset and dataloader
    train_set = MCDataset(args, tokenizer, mode='train')
    valid_set = MCDataset(args, tokenizer, mode='valid')
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_set, collate_fn=valid_set.collate_fn, shuffle=False, batch_size=args.batch_size)

    # Model
    model = MCModel(args, config)
    if args.wandb:
        wandb.watch(model)

    # Optimizer and Learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    # warmup_step = int(0.1 * len(train_loader)) // args.accumulation_steps
    warmup_step = args.warmup_step
    total_step = args.num_epoch * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)

    start_epoch = 1
    if args.resume is not None:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        args.model_name = state['name']
        scheduler = state['scheduler']

    # Use Accelerator
    accelerator = Accelerator(fp16=True)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

    for epoch in range(start_epoch, args.num_epoch + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_acc = train(model, accelerator, train_loader, optimizer, args, scheduler)
        valid_loss, valid_acc = validate(model, valid_loader)
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}")
        print(f"Valid Accuracy: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}")
        if args.wandb:
            wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss, "Validation Accuracy": valid_acc, "Validation Loss": valid_loss,})
        save_model(model, epoch, optimizer, scheduler, args)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the cache file.",
        default="./cache",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Directory to save the model file.",
        default=None,
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        help="Directory to save the model file.",
        default=None,
    )
    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--accumulation_steps", type=int, default=10)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--warmup_step", type=int, default=30)
    parser.add_argument("--task", type=str, default="MC")
    parser.add_argument("--resume", type=Path, default=None)

    args = parser.parse_args()
    return args
    """
    parameters = {
        "data_dir": Path("/content//data/"),
        "cache_dir": Path("/content/cache/"),
        "model_name": "hfl/chinese-macbert-base",
        "ckpt_dir": Path("/content/ckpt/"),
        "context_path": None,
        "json_path": None,
        
        "max_len": 512,
        "lr": 2e-5,
        "weight_decay": 1e-6,
        "batch_size": 4,
        "device": "cuda",
        "num_epoch": 5,
        "accumulation_steps": 10,
        "prefix": "",
        "wandb": False,
        "from_scratch": False,
        "warmup_step": 30,

        "seed": 1126,

        "task": "MC",

        "resume": None,
    }
    config = Namespace(**parameters)
    return config
    """


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # wandb_config = vars(args)
    # run = wandb.init(
    #     project = f"ADL_HW2",
    #     config = wandb_config,
    #     reinit = True,
    #     group = "Multi_Choise",
    #     resume = "allow"
    # )
    # artifact = wandb.Artifact("model", type="model")
    if args.wandb:
        wandb.login()
        wandb.init(
            project="fADL_HW2",
            group = "Multi_Choise",
        )
        wandb.config.update(args)
    main(args)