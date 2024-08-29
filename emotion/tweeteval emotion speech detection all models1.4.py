#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:47:31 2024

@author: saiful
"""

# -*- coding: utf-8 -*-
"""x1_FakeNewDetection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KPShsnIudbmI_y-brdDEcdPYLQSbtZqQ

## Setup Environment
"""
# https://github.com/cardiffnlp/tweeteval
# https://github.com/Vicomtech/hate-speech-dataset
# https://www.kaggle.com/datasets/thedevastator/tweeteval-a-multi-task-classification-benchmark?select=emotion_train.csv
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
from my_models import get_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Load Dataset"""

# Load the text file using pandas
emotion_train = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/emotion_train.csv")
# Display the first few rows of the dataframe
print(emotion_train.head())

# Separate the dataframe into two dataframes
train_text = emotion_train[['text']].copy()  # Retain the 'text' column with the header
train_labels = emotion_train[['label']].copy()  # Retain the 'label' column with the header

# Display the first few rows of each dataframe
print(train_text.head())
print(train_labels.head())

# val
# Load the text file using pandas
emotion_validation = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/emotion_validation.csv")
# Display the first few rows of the dataframe
print(emotion_validation.head())

# Separate the dataframe into two dataframes
val_text = emotion_validation[['text']].copy()  # Retain the 'text' column with the header
val_labels = emotion_validation[['label']].copy()  # Retain the 'label' column with the header

# Display the first few rows of each dataframe
print(val_text.head())
print(val_labels.head())

## ## test
# Load the text file using pandas
emotion_test = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/emotion_test.csv")
# Display the first few rows of the dataframe
print(emotion_test.head())

# Separate the dataframe into two dataframes
test_text = emotion_test[['text']].copy()  # Retain the 'text' column with the header
test_labels = emotion_test[['label']].copy()  # Retain the 'label' column with the header

# Display the first few rows of each dataframe
print(test_text.head())
print(test_text.head())


train_text =train_text.squeeze()
val_text=val_text.squeeze()
test_text=test_text.squeeze()

train_labels =train_labels.squeeze()
val_labels =val_labels.squeeze()
test_labels =test_labels.squeeze()

# Convert each item inside the series to a string
train_text = train_text.apply(str)
val_text = val_text.apply(str)
test_text = test_text.apply(str)

#%%
# # Checking if our data is well balanced
# label_size = [data['label'].sum(), len(data['label']) - data['label'].sum()]
# plt.pie(label_size, explode=[0.1, 0.1], colors=['firebrick', 'navy'], startangle=90, shadow=True, labels=['Fake', 'True'], autopct='%1.1f%%')

# """## Train-test-split"""

# # Train-Validation-Test set split into 70:15:15 ratio
# # Train-Temp split
# train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],
#                                                                     random_state=2018,
#                                                                     test_size=0.3,
#                                                                     stratify=data['label'])
# # Validation-Test split
# val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
#                                                                 random_state=2018,
#                                                                 test_size=0.5,
#                                                                 stratify=temp_labels)
#%% model
"""## BERT Fine-tuning

### Load pretrained BERT Model
"""
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2Model, GPT2TokenizerFast, GPT2ForSequenceClassification
from transformers import LongformerModel, LongformerTokenizerFast
from transformers import LukeModel, LukeTokenizer
from transformers import T5Model, T5Tokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import ElectraModel, ElectraTokenizer


"""### Prepare Input Data"""
#plot
# # Plot histogram of the number of words in train data 'title'
# seq_len = [len(title.split()) for title in train_text]

# pd.Series(seq_len).hist(bins=40, color='firebrick')
# plt.xlabel('Number of Words')
# plt.ylabel('Number of texts')

# model , tokenizer = get_model(called_model ="bert" )
# model , tokenizer = get_model(called_model ="roberta" )
# model , tokenizer = get_model(called_model ="distilbert" )
# model , tokenizer = get_model(called_model ="electra" )
# model , tokenizer = get_model(called_model ="gpt2" )
# model , tokenizer = get_model(called_model ="longformer" )
# model , tokenizer = get_model(called_model ="luke" )
model , tokenizer = get_model(called_model ="t5" )
# model , tokenizer = get_model(called_model ="xlnet" )



#%
# BERT Tokeizer Functionality
sample_data = ["Build fake news model.",
               "Using bert."]  # sample data
tokenized_sample_data = tokenizer.batch_encode_plus(sample_data, padding=True)  # encode text
print(tokenized_sample_data)
# Majority of titles above have word length under 15. So, we set max title length as 15
MAX_LENGTH = 100

def tokenize_and_encode_sequences(tokenizer):
    

    # Tokenize and encode sequences in the train set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    
    # Convert lists to tensors
    train_seq = torch.tensor(tokens_train['input_ids']).to(device)
    train_mask = torch.tensor(tokens_train['attention_mask']).to(device)
    train_y = torch.tensor(train_labels.tolist()).to(device)
    
    val_seq = torch.tensor(tokens_val['input_ids']).to(device)
    val_mask = torch.tensor(tokens_val['attention_mask']).to(device)
    val_y = torch.tensor(val_labels.tolist()).to(device)
    
    test_seq = torch.tensor(tokens_test['input_ids']).to(device)
    test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
    test_y = torch.tensor(test_labels.tolist()).to(device)
    
    # Data Loader structure definition
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    batch_size = 32  # define a batch size
    
    train_data = TensorDataset(train_seq, train_mask, train_y)  # wrap tensors
    train_sampler = RandomSampler(train_data)  # sampler for sampling the data during training
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)  # dataLoader for train set
    
    val_data = TensorDataset(val_seq, val_mask, val_y)  # wrap tensors
    val_sampler = SequentialSampler(val_data)  # sampler for sampling the data during training
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)  # dataLoader for validation set
    
    
    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    
    return train_dataloader, val_dataloader,test_dataloader



#%%
"""### Freeze Layers"""


# Defining the hyperparameters (optimizer, weights of the classes and the epochs)
# from transformers import AdamW, Adam
from torch.optim import Adam, AdamW
optimizer = AdamW(model.parameters(), lr=1e-4)  # learning rate
# cross_entropy = nn.NLLLoss()  # Define the loss function
cross_entropy = nn.CrossEntropyLoss()  # Define the loss function

epochs = 3  # Number of training epochs

#%%
"""### Define Train & Evaluate Function"""

# Defining training and evaluation functions
def train(train_dataloader,model ):
    model.train()
    total_loss, total_accuracy = 0, 0

    for step, batch in enumerate(train_dataloader):  # iterate over batches
        if step % 50 == 0 and not step == 0:  # progress update after every 50 batches.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]  # push the batch to gpu
        sent_id, mask, labels = batch
        model.zero_grad()  # clear previously calculated gradients
        preds = model(sent_id, mask)  # get model predictions for current batch
        loss = cross_entropy(preds, labels.long())  # compute loss between actual & predicted values
        total_loss = total_loss + loss.item()  # add on to the total loss
        loss.backward()  # backward pass to calculate the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients to 1.0. It helps in preventing exploding gradient problem
        optimizer.step()  # update parameters

    avg_loss = total_loss / len(train_dataloader)  # compute training loss of the epoch
    return avg_loss  # returns the loss

def evaluate(dataloader):
    print("\nEvaluating...")
    model.eval()  # Deactivate dropout layers
    total_loss, total_accuracy = 0, 0
    total_preds, total_labels = [], []

    for step, batch in enumerate(dataloader):  # Iterate over batches
        if step % 50 == 0 and not step == 0:  # Progress update every 50 batches.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        batch = [t.to(device) for t in batch]  # Push the batch to GPU
        sent_id, mask, labels = batch
        with torch.no_grad():  # Deactivate autograd
            preds = model(sent_id, mask)  # Model predictions
            loss = cross_entropy(preds, labels.long())  # Compute the validation loss between actual and predicted values
            total_loss += loss.item()

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_preds.extend(np.argmax(preds, axis=1))
            total_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)  # Compute the validation loss of the epoch
    total_accuracy = np.sum(np.array(total_preds) == np.array(total_labels)) / len(total_labels)  # Compute accuracy

    return avg_loss, total_accuracy

def train_validate_and_test(train_dataloader, val_dataloader, test_dataloader):
    # Train and predict
    best_valid_loss = float('inf')
    train_losses = []  # Empty lists to store training and validation loss of each epoch
    valid_losses = []
    valid_accuracies = []
    
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        # Train model
        train_loss = train(train_dataloader,model)
        # Evaluate model
        valid_loss, valid_accuracy = evaluate(val_dataloader)
        
        if valid_loss < best_valid_loss:  # Save the best model
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'c2_new_model_weights.pt')
    
        train_losses.append(train_loss)  # Append training and validation loss
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
    
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'Validation Accuracy: {valid_accuracy:.3f}')
    
    
    
    # Load weights of the best model
    path = 'c2_new_model_weights.pt'
    model.load_state_dict(torch.load(path))
    
    # Evaluate the model on the test set
    def test_model():
        model.eval()  # Set the model to evaluation mode
        total_preds, total_labels = [], []
    
        for batch in test_dataloader:
            batch = [t.to(device) for t in batch]  # Push the batch to GPU
            sent_id, mask, labels = batch
            with torch.no_grad():
                preds = model(sent_id, mask)
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                total_preds.extend(np.argmax(preds, axis=1))
                total_labels.extend(labels)
        
        total_preds = np.array(total_preds)
        total_labels = np.array(total_labels)
        
        test_accuracy = np.sum(total_preds == total_labels) / len(total_labels)
        return test_accuracy, total_labels, total_preds
    
    test_accuracy, total_labels, total_preds = test_model()
    print("\nTest Accuracy: {:.3f}".format(test_accuracy))
    print(classification_report(total_labels, total_preds))
    
def main_function(called_model ="t5"):
    print("\n\n\n===================================================== ")
    print(f"flag 1.10  model: {called_model} has started  ", called_model)
    print("===================================================== ")

    model , tokenizer = get_model(called_model ="t5" )
    train_dataloader, val_dataloader, test_dataloader = tokenize_and_encode_sequences(tokenizer)
    train_validate_and_test(train_dataloader, val_dataloader, test_dataloader)
    print(f"flag 1.10  model: {called_model} has finished  :", called_model)

    
main_function(called_model ="bert")
main_function(called_model ="roberta")
main_function(called_model ="distilbert")
main_function(called_model ="electra")
main_function(called_model ="gpt2")
main_function(called_model ="longformer")
main_function(called_model ="luke")
main_function(called_model ="t5")
main_function(called_model ="xlnet")