import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup

from datasets import QADataset
from model import QAModel
from utils import same_seeds, save_model, get_config

def train(accelerator, args, data_loader, model, optimizer, scheduler=None):
    model.train()
    train_loss, train_acc = [], []

    for idx, batch in enumerate(tqdm(data_loader)):
        ids, inputs = batch
        """
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        """
        qa_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            start_positions=inputs["start_positions"],
            end_positions=inputs["end_positions"],
        )

        loss = qa_output.loss
        loss = loss / args.accumulation_steps
        accelerator.backward(loss)

        start_logits = qa_output.start_logits.argmax(dim=-1)
        end_logits = qa_output.end_logits.argmax(dim=-1)
        acc = ((inputs["start_positions"] == start_logits) & (inputs["end_positions"] == end_logits)).cpu().numpy().mean()

        if ((idx + 1) % args.accumulation_steps == 0) or (idx == len(data_loader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        train_loss.append(loss.item())
        train_acc.append(acc)

    return sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc)


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    valid_loss, valid_acc = [], []

    for batch in tqdm(data_loader):
        ids, inputs = batch
        """
        n = inputs["input_ids"].shape[0]
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        """
        qa_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            start_positions=inputs["start_positions"],
            end_positions=inputs["end_positions"],
        )
        loss = qa_output.loss
        start_logits = qa_output.start_logits.argmax(dim=-1)
        end_logits = qa_output.end_logits.argmax(dim=-1)
        acc = ((inputs["start_positions"] == start_logits) & (inputs["end_positions"] == end_logits)).cpu().numpy().mean()

        valid_loss.append(loss.item())
        valid_acc.append(acc)

    return sum(valid_loss) / len(valid_loss), sum(valid_acc) / len(valid_acc)


def main(args):
    same_seeds(args.seed)
    
    if args.from_scratch:
        config = get_config()
    else:
        config = AutoConfig.from_pretrained(args.model_name)
    print(config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=config, model_max_length=args.max_len, use_fast=True)

    train_set = QADataset(args, tokenizer, mode='train')
    valid_set = QADataset(args, tokenizer, mode='valid')
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, shuffle=True, batch_size=args.batch_size,)
    valid_loader = DataLoader(valid_set, collate_fn=valid_set.collate_fn, shuffle=False, batch_size=args.batch_size,)

    model = QAModel(args, config)
    if args.wandb:
        wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    warmup_step = args.warmup_step
    total_step = args.num_epoch * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)


    start_epoch = 1
    if args.resume is not None:
        state = torch.load(args.resume)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        args.model_name = state['name']
        scheduler = state['scheduler']

    accelerator = Accelerator(fp16=True)
    print(f"Using {accelerator.device}")
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

    best_loss = float("inf")

    for epoch in range(start_epoch, args.num_epoch + 1):
        train_loss, train_acc = train(accelerator, args, train_loader, model, optimizer, scheduler)
        valid_loss, valid_acc = validate(valid_loader, model)
        print(f"Epoch {epoch}:")
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}")
        print(f"Valid Accuracy: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}")
        if args.wandb:
            wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss, "Validation Accuracy": valid_acc, "Validation Loss": valid_loss,})
        """
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_model(model, epoch, optimizer, scheduler, args)
            
            torch.save(
                {
                    "name": args.model_name,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"{args.prefix}qa_loss.ckpt"),
            )
            """
        save_model(model, epoch, optimizer, scheduler, args)
        """
        torch.save(
            {
                "name": args.model_name,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.ckpt_dir, f"{args.prefix}qa_{epoch}.ckpt"),
        )
        """

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default=".",
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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--warmup_step", type=int, default=30)
    parser.add_argument("--task", type=str, default="QA")
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
        "lr": 3e-5,
        "weight_decay": 1e-6,
        "batch_size": 16,
        "device": "cuda",
        "num_epoch": 5,
        "accumulation_steps": 4,
        "prefix": "",
        "wandb": False,
        "from_scratch": False,
        "warmup_step": 30,
        # "load": None,

        "seed": 1126,

        "task": "QA",

        "resume": None,
    }
    config = Namespace(**parameters)
    return config"""


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    wandb_config = vars(args)
    run = wandb.init(
        project = f"ADL_HW2",
        config = wandb_config,
        reinit = True,
        group = "Questions_Answering",
        resume = "allow"
    )
    artifact = wandb.Artifact("model", type="model")
    main(args)