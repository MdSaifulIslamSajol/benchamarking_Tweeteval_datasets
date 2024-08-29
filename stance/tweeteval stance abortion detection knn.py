import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
stance_abortion_train = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/stance_abortion_train.csv")
stance_abortion_validation = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/stance_abortion_validation.csv")
stance_abortion_test = pd.read_csv("/home/saiful/bangla fault news/tweeteval/tweeteval kaggle/stance_abortion_test.csv")

# Separate the datasets into text and labels
train_text = stance_abortion_train['text'].apply(str)
train_labels = stance_abortion_train['label']
val_text = stance_abortion_validation['text'].apply(str)
val_labels = stance_abortion_validation['label']
test_text = stance_abortion_test['text'].apply(str)
test_labels = stance_abortion_test['label']

# Load the RoBERTa model and tokenizer
model = AutoModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

MAX_LENGTH = 150

# Tokenize and encode sequences
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True
)
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True
)
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True
)

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids']).to(device)
train_mask = torch.tensor(tokens_train['attention_mask']).to(device)
train_y = torch.tensor(train_labels.values).to(device)

val_seq = torch.tensor(tokens_val['input_ids']).to(device)
val_mask = torch.tensor(tokens_val['attention_mask']).to(device)
val_y = torch.tensor(val_labels.values).to(device)

test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.values).to(device)

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = BERT_Arch(model).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
cross_entropy = nn.NLLLoss()

epochs = 10

# Training function
def train():
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels.long())
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

# Evaluation function
def evaluate():
    model.eval()
    total_loss = 0
    total_preds, total_labels = [], []

    for step, batch in enumerate(val_dataloader):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels.long())
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_preds.extend(np.argmax(preds, axis=1))
            total_labels.extend(labels)

    avg_loss = total_loss / len(val_dataloader)
    total_accuracy = np.sum(np.array(total_preds) == np.array(total_labels)) / len(total_labels)

    return avg_loss, total_accuracy

# Data Loader structure definition
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=val_sampler, batch_size=batch_size)

# Train and evaluate the model
best_valid_loss = float('inf')
train_losses = []
valid_losses = []
valid_accuracies = []

for epoch in range(epochs):
    print(f'\n Epoch {epoch + 1} / {epochs}')
    train_loss = train()
    valid_loss, valid_accuracy = evaluate()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'c2_new_model_weights.pt')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    print(f'Validation Accuracy: {valid_accuracy:.3f}')

# Load the best model weights
model.load_state_dict(torch.load('c2_new_model_weights.pt'))

# Extract embeddings using the trained BERT model
def extract_embeddings(dataloader):
    model.eval()
    embeddings = []

    for step, batch in enumerate(dataloader):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            cls_hs = model.bert(sent_id, attention_mask=mask)['pooler_output']
            embeddings.append(cls_hs.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

train_embeddings = extract_embeddings(train_dataloader)
val_embeddings = extract_embeddings(val_dataloader)
test_embeddings = extract_embeddings(test_dataloader)

# Scale the embeddings
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
val_embeddings = scaler.transform(val_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)

# Evaluate KNN classifier
val_preds = knn.predict(val_embeddings)
test_preds = knn.predict(test_embeddings)

# Print classification reports
print("Validation Classification Report:")
print(classification_report(val_labels, val_preds))

print("Test Classification Report:")
print(classification_report(test_labels, test_preds))
