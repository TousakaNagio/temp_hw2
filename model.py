from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering


class MCModel(nn.Module):
    def __init__(self, args, config, namae=None):
        super(MCModel, self).__init__()
        self.name = args.model_name
        if args.from_scratch:
            self.model = AutoModelForMultipleChoice.from_config(config)
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(self.name, config = config)
        """
        try:
            if args.from_scratch:
                self.model = AutoModelForMultipleChoice.from_config(config)
            else:
                self.model = AutoModelForMultipleChoice.from_pretrained(self.name, config = config)
        except:
            self.model = AutoModelForMultipleChoice.from_config(config)
        """

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False

class QAModel(nn.Module):
    def __init__(self, args, config, namae=None):
        super(QAModel, self).__init__()
        self.name = args.model_name
        if args.from_scratch:
            self.model = AutoModelForQuestionAnswering.from_config(config)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.name, config = config)
        """
        self.name = namae if namae is not None else args.model_name
        try:
            if args.from_scratch:
                self.model = AutoModelForQuestionAnswering.from_config(config)
            else:
                self.model = AutoModelForQuestionAnswering.from_pretrain(config = config)
        except:
            self.model = AutoModelForQuestionAnswering.from_config(config)
        """
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False