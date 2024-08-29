#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:42:26 2024

@author: saiful
"""
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import  DistilBertModel, DistilBertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2Model, GPT2TokenizerFast, GPT2ForSequenceClassification
from transformers import LongformerModel, LongformerTokenizerFast
from transformers import LukeModel, LukeTokenizer
from transformers import T5Model, T5Tokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import ElectraModel, ElectraTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to control the requires_grad setting
TRAINABLE = True
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']  # pass the inputs to the model
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

class RoBERT_Arch(nn.Module):
    def __init__(self, bert):
        super(RoBERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']  # pass the inputs to the model
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = BERT_Arch(model).to(device)

# Define the DistilBERT architecture
class DistilBERT_Arch(nn.Module):
    def __init__(self, bert):
        super(DistilBERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        cls_hs = self.bert(input_ids=sent_id, attention_mask=mask).last_hidden_state[:, 0, :]  # use the hidden state of the [CLS] token
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = DistilBERT_Arch(model).to(device)


# Define the GPT2 architecture for sequence classification
class GPT2_Arch(nn.Module):
    def __init__(self, gpt_model):
        super(GPT2_Arch, self).__init__()
        self.gpt = gpt_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        gpt_output = self.gpt(input_ids=sent_id, attention_mask=mask)
        cls_hs = gpt_output.last_hidden_state[:, -1, :]  # use the hidden state of the last token
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x
# model = GPT2_Arch(model).to(device)

# Define the Longformer architecture for sequence classification
class Longformer_Arch(nn.Module):
    def __init__(self, longformer_model):
        super(Longformer_Arch, self).__init__()
        self.longformer = longformer_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        longformer_output = self.longformer(input_ids=sent_id, attention_mask=mask)
        cls_hs = longformer_output.last_hidden_state[:, 0, :]  # use the hidden state of the first token ([CLS] token)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = Longformer_Arch(model).to(device)

# Define the LUKE architecture for sequence classification
class LUKE_Arch(nn.Module):
    def __init__(self, luke_model):
        super(LUKE_Arch, self).__init__()
        self.luke = luke_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        luke_output = self.luke(input_ids=sent_id, attention_mask=mask)
        cls_hs = luke_output.last_hidden_state[:, 0, :]  # use the hidden state of the first token ([CLS] token)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = LUKE_Arch(model).to(device)

# Define the T5 architecture for sequence classification
class T5_Arch(nn.Module):
    def __init__(self, t5_model):
        super(T5_Arch, self).__init__()
        self.t5 = t5_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        decoder_input_ids = torch.zeros(sent_id.shape, dtype=torch.long).to(device)
        t5_output = self.t5(input_ids=sent_id, attention_mask=mask, decoder_input_ids=decoder_input_ids)
        cls_hs = t5_output.last_hidden_state[:, 0, :]  # use the hidden state of the first token ([CLS] token)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = T5_Arch(model).to(device)

# Define the XLNet architecture for sequence classification
class XLNet_Arch(nn.Module):
    def __init__(self, xlnet_model):
        super(XLNet_Arch, self).__init__()
        self.xlnet = xlnet_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        xlnet_output = self.xlnet(input_ids=sent_id, attention_mask=mask)
        cls_hs = xlnet_output.last_hidden_state[:, -1, :]  # use the hidden state of the last token
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = XLNet_Arch(model).to(device)

# Define the ELECTRA architecture for sequence classification
class ELECTRA_Arch(nn.Module):
    def __init__(self, electra_model):
        super(ELECTRA_Arch, self).__init__()
        self.electra = electra_model
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 4)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        electra_output = self.electra(input_ids=sent_id, attention_mask=mask)
        cls_hs = electra_output.last_hidden_state[:, 0, :]  # use the hidden state of the first token ([CLS] token)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

# model = ELECTRA_Arch(model).to(device)


def set_trainable(model, requires_grad=TRAINABLE):
    """
    Set the requires_grad attribute of the model parameters and print their status.

    Args:
    model (torch.nn.Module): The model whose parameters are to be modified.
    requires_grad (bool): Whether the parameters require gradients.
    """
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
        # print(f"{name}: requires_grad={param.requires_grad}")

def get_model(called_model ="bert" ):

    if(called_model== "bert"):
        # # # # Load BERT model and tokenizer via HuggingFace Transformers
        model_name = "bert-base-uncased"
        model = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', legacy=False )
        
        set_trainable(model, requires_grad=TRAINABLE)
        model = BERT_Arch(model).to(device)


    elif(called_model== "roberta"):
        # # Load the RoBERTa model and tokenizer
        model_name = "roberta-base"
        model = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        set_trainable(model, requires_grad=TRAINABLE)
        model = RoBERT_Arch(model).to(device)
    
  
    elif(called_model== "distilbert"):
        # Load the DistilBERT model and tokenizer
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        set_trainable(model, requires_grad=TRAINABLE)
        model = DistilBERT_Arch(model).to(device)

    
    elif(called_model== "electra"):

        # Load the ELECTRA model and tokenizer
        model_name = "google/electra-base-discriminator"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        model = ElectraModel.from_pretrained(model_name)
        
        set_trainable(model, requires_grad=TRAINABLE)
        model = ELECTRA_Arch(model).to(device)


    elif(called_model== "gpt2"):

        # Load the GPT-2 model and tokenizer
        model_name = "gpt2"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2Model.from_pretrained(model_name)
        # Add padding token to tokenizer and resize model embeddings
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
        set_trainable(model, requires_grad=TRAINABLE)
        model = GPT2_Arch(model).to(device)

    
    elif(called_model== "longformer"):

        # Load the Longformer model and tokenizer
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        model = LongformerModel.from_pretrained(model_name)
    
        set_trainable(model, requires_grad=TRAINABLE)
        model = Longformer_Arch(model).to(device)

    
    elif(called_model== "luke"):

        # Load the LUKE model and tokenizer
        model_name = "studio-ousia/luke-base"
        tokenizer = LukeTokenizer.from_pretrained(model_name)
        model = LukeModel.from_pretrained(model_name)
    
        set_trainable(model, requires_grad=TRAINABLE)
        model = LUKE_Arch(model).to(device)

    elif(called_model== "t5"):

        # Load the T5 model and tokenizer
        model_name = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5Model.from_pretrained(model_name)
        # Add padding token to tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        MAX_LENGTH = 100  # T5's max length
    
        set_trainable(model, requires_grad=TRAINABLE)
        model = T5_Arch(model).to(device)

    elif(called_model== "xlnet"):

        # Load the XLNet model and tokenizer
        model_name = "xlnet-base-cased"
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        model = XLNetModel.from_pretrained(model_name)
        # Add padding token to tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
        set_trainable(model, requires_grad=TRAINABLE)
        model = XLNet_Arch(model).to(device)

    return model, tokenizer
