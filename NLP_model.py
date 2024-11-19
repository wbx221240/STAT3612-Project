import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns  # Visualisation
import matplotlib.pyplot as plt  # Visualisation
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torchmetrics import Accuracy, AUROC, F1Score
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_df = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/train.csv")
notes_df = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/notes.csv")

# Merge notes with training and validation data based on ID
merged_train = train_df.merge(notes_df, on='id', how='left')

# Load clinicalBERT tokenizer and model, and move the model to GPU
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Preprocessing function
MAX_LENGTH = 512
def preprocess_function(batch_texts):
    """Tokenize a batch of texts."""
    return tokenizer(batch_texts, 
                     truncation=True, 
                     max_length=MAX_LENGTH, 
                     padding='max_length',
                     return_tensors="pt")

# Tokenize datasets in batches
def tokenize_in_batches(texts, batch_size=100):
    """Tokenize large datasets in smaller batches and move to GPU."""
    tokenized_batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokenized_batch = preprocess_function(batch_texts.tolist())
        # Move tokenized data to GPU
        tokenized_batch = {key: value.to(device) for key, value in tokenized_batch.items()}
        tokenized_batches.append(tokenized_batch)
    return tokenized_batches

# Tokenize merged training and validation datasets
batch_size = 100  # Adjust as necessary based on available memory
tokenized_train = tokenize_in_batches(merged_train['text'], batch_size=batch_size)

# Check the size of tokenized outputs (for debugging)
print(f"Number of training batches: {len(tokenized_train)}")

# Define a function to extract embeddings using clinicalBERT
def extract_embeddings(tokenized_batches):
    """Extract embeddings for tokenized batches using clinicalBERT."""
    embeddings = []
    with torch.no_grad():
        for batch in tokenized_batches:
            outputs = model(**batch)
            # Use [CLS] token embeddings from the last hidden layer
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            embeddings.append(cls_embeddings.cpu().numpy())  # Move to CPU and convert to numpy
    return np.concatenate(embeddings, axis=0)  # Concatenate all batches

# Extract embeddings for training and validation datasets
print("Extracting embeddings for training data...")
train_embeddings = extract_embeddings(tokenized_train)

# Save embeddings to disk for later use
os.makedirs("embeddings", exist_ok=True)
train_emb_path = "embeddings/train_embeddings.npy"
np.save(train_emb_path, train_embeddings)

print(f"Training embeddings saved to: {train_emb_path}")

# Load labels
train_labels = merged_train['readmitted_within_30days'].values
np.save("embeddings/train_labels.npy", train_labels)
# Debugging: Check dimensions
print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Train labels shape: {train_labels.shape}")



#MLP
# 将特征和标签保存为 NumPy 数组（已保存到磁盘，这里直接加载）
train_embeddings = np.load("embeddings/train_embeddings.npy")
train_labels = np.load("embeddings/train_labels.npy")

# 确保特征和标签是 NumPy 数组
assert train_embeddings.shape[0] == train_labels.shape[0], "Features and labels must have the same number of samples."

# 自定义数据集类
class ReadmissionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建数据集和数据加载器
dataset = ReadmissionDataset(train_embeddings, train_labels)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

accuracy = Accuracy(task="binary")
auroc_metric = AUROC(task="binary")
f1_metric = F1Score(task="binary")

nlp_net = nn.Sequential(
    nn.Linear(768,256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(64,8),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.BatchNorm1d(8),
    nn.Linear(8,1)    
)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

nlp_net.apply(init_weights)

# 损失函数和优化器
loss = nn.BCEWithLogitsLoss()
trainer = torch.optim.AdamW(nlp_net.parameters(), lr=0.01, weight_decay=1e-4)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    nlp_net.train()  # 设置为训练模式
    running_loss = 0.0
    total_accuracy = 0.0
    total_auc = 0.0
    total_f1 = 0.0
    for i, (features, labels) in enumerate(train_loader):
        # 计算输出
        logits = nlp_net(features).squeeze()  # 得到 logits，形状为 [batch_size]
        batch_loss = loss(logits, labels)
        trainer.zero_grad()
        batch_loss.backward()
        trainer.step()
        
        running_loss += batch_loss.item()
        
        probabilities = torch.sigmoid(logits)
        batch_accuracy = accuracy(probabilities, labels)
        batch_auc = auroc_metric(probabilities, labels)
        batch_f1 = f1_metric(probabilities, labels)

        total_accuracy += batch_accuracy
        total_auc += batch_auc
        total_f1 += batch_f1


    # 输出每个 epoch 的平均 loss, accuracy 和 AUC
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_auc = total_auc / len(train_loader)
    avg_f1 = total_f1/len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1:{avg_f1}")
