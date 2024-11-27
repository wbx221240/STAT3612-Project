import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns  # Visualisation
import matplotlib.pyplot as plt  # Visualisation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from torchmetrics import Accuracy, AUROC, F1Score

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_df = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/train.csv")
test_df = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/valid.csv")
notes_df = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/notes.csv")

# Merge notes with training and validation data based on ID
merged_train = train_df.merge(notes_df, on='id', how='left')

# TF-IDF Vectorization for Training Set
print("Extracting TF-IDF features for training set...")
tfidf_vectorizer = TfidfVectorizer(max_features=768)  # Adjust `max_features` as needed
train_tfidf_features = tfidf_vectorizer.fit_transform(merged_train['text']).toarray()
train_labels = merged_train['readmitted_within_30days'].values

# Save training features and labels to disk (optional)
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/train_tfidf_features.npy", train_tfidf_features)
np.save("embeddings/train_labels.npy", train_labels)

# TF-IDF Vectorization for Test Set
print("Extracting TF-IDF features for test set...")
merged_test = test_df.merge(notes_df, on='id', how='left')
test_tfidf_features = tfidf_vectorizer.transform(merged_test['text']).toarray()
test_labels = merged_test['readmitted_within_30days'].values

# Save test features and labels to disk (optional)
np.save("embeddings/test_tfidf_features.npy", test_tfidf_features)
np.save("embeddings/test_labels.npy", test_labels)

# Custom Dataset Class
class ReadmissionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create Datasets and DataLoaders
train_dataset = ReadmissionDataset(train_tfidf_features, train_labels)
test_dataset = ReadmissionDataset(test_tfidf_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Metrics
accuracy = Accuracy(task="binary").to(device)
auroc_metric = AUROC(task="binary").to(device)
f1_metric = F1Score(task="binary").to(device)

# Define MLP Model
nlp_net = nn.Sequential(
    nn.Linear(768, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(64, 8),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.BatchNorm1d(8),
    nn.Linear(8, 1)
).to(device)

# Initialize Weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

nlp_net.apply(init_weights)

# Loss Function and Optimizer
loss = nn.BCEWithLogitsLoss()
trainer = torch.optim.AdamW(nlp_net.parameters(), lr=0.01, weight_decay=1e-4)

# Training and Validation Loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training Phase
    nlp_net.train()
    running_loss = 0.0
    total_accuracy = 0.0
    total_auc = 0.0
    total_f1 = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        logits = nlp_net(features).squeeze()
        batch_loss = loss(logits, labels)
        trainer.zero_grad()
        batch_loss.backward()
        trainer.step()

        running_loss += batch_loss.item()
        probabilities = torch.sigmoid(logits)
        total_accuracy += accuracy(probabilities, labels)
        total_auc += auroc_metric(probabilities, labels)
        total_f1 += f1_metric(probabilities, labels)

    avg_loss = running_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_auc = total_auc / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Train AUC: {avg_auc:.4f}, Train F1: {avg_f1:.4f}")

    # Validation Phase
    nlp_net.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_auc = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = nlp_net(features).squeeze()
            batch_loss = loss(logits, labels)
            val_loss += batch_loss.item()

            probabilities = torch.sigmoid(logits)
            val_accuracy += accuracy(probabilities, labels)
            val_auc += auroc_metric(probabilities, labels)
            val_f1 += f1_metric(probabilities, labels)

    avg_val_loss = val_loss / len(test_loader)
    avg_val_accuracy = val_accuracy / len(test_loader)
    avg_val_auc = val_auc / len(test_loader)
    avg_val_f1 = val_f1 / len(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_val_loss:.4f}, Test Accuracy: {avg_val_accuracy:.4f}, Test AUC: {avg_val_auc:.4f}, Test F1: {avg_val_f1:.4f}")
