import random

import numpy as np
import torch
from transformers import BertConfig


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(model, epoch, optimizer, scheduler, args):
    file_name = f"{args.prefix}_{args.task}_{epoch}.ckpt"
    torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "name": args.model_name,
            }, os.path.join(args.ckpt_dir, file_name))

def get_config():
    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=512,
        classifier_dropout=0.3,
        pooler_fc_size=256,
        pooler_num_attention_heads=2,
        return_dict=False,
    )
    return config