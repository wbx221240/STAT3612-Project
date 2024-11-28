import torch
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torchmetrics import Accuracy, AUROC, F1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化指标，并将它们移到相同的设备上
accuracy = Accuracy(task="binary").to(device)
auroc_metric = AUROC(task="binary").to(device)
f1_metric = F1Score(task="binary").to(device)

# 读取训练集数据
train_set = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/train.csv")

train_set['readmitted_within_30days'] = train_set['readmitted_within_30days'].replace({'True': 1, 'False': 0})
image_path = train_set['image_path']
image_label = train_set['readmitted_within_30days']
folder_images = os.listdir('C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features')

# 匹配文件和标签
matched_files = {}
for path, label in zip(image_path, image_label):
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    full_path = os.path.join(
        'C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features', 
        path
    )
    if path in folder_images:
        matched_files[full_path] = label

# 加载验证集数据
valid_set = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/valid.csv")
valid_set = valid_set[valid_set['ViewPosition'] == 'AP']
valid_set['readmitted_within_30days'] = valid_set['readmitted_within_30days'].replace({'True': 1, 'False': 0})
valid_image_path = valid_set['image_path']
valid_image_label = valid_set['readmitted_within_30days']

valid_matched_files = {}
for path, label in zip(valid_image_path, valid_image_label):
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    full_path = os.path.join(
        'C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features', 
        path
    )
    if path in folder_images:
        valid_matched_files[full_path] = label

# 自定义数据集类
class ReadmissionDataset(Dataset):
    def __init__(self, matched_files):
        self.matched_files = matched_files
        self.image_paths = list(matched_files.keys())
        self.labels = list(matched_files.values())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        with open(image_path, 'rb') as f:
            image_features = pickle.load(f)
        return torch.tensor(image_features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 验证集数据加载器
valid_dataset = ReadmissionDataset(valid_matched_files)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

# 模型定义
image_net = nn.Sequential(
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(128, 32),
    nn.BatchNorm1d(32),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(32, 1)
).to(device)  # 将模型移动到设备上

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

# 使用训练集训练模型，验证集评估
num_epochs = 10
trainer = torch.optim.AdamW(image_net.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(num_epochs):
    image_net.train()
    # 使用训练集进行训练
    for features, labels in valid_loader:  # 使用valid_loader进行训练（如果你只想使用验证集数据训练）
        features, labels = features.to(device), labels.to(device)
        logits = image_net(features).squeeze()
        batch_loss = nn.BCEWithLogitsLoss()(logits, labels)
        trainer.zero_grad()
        batch_loss.backward()
        trainer.step()

    # 验证集评估
    image_net.eval()
    valid_running_loss = 0.0
    valid_accuracy = 0.0
    valid_auc = 0.0
    valid_f1 = 0.0

    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), labels.to(device)
            logits = image_net(features).squeeze()
            probabilities = torch.sigmoid(logits)

            valid_running_loss += batch_loss.item()
            valid_accuracy += accuracy(probabilities, labels)
            valid_auc += auroc_metric(probabilities, labels)
            valid_f1 += f1_metric(probabilities, labels)

    avg_loss = valid_running_loss / len(valid_loader)
    avg_accuracy = valid_accuracy / len(valid_loader)
    avg_auc = valid_auc / len(valid_loader)
    avg_f1 = valid_f1 / len(valid_loader)

    print(f"Validation Results for Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1: {avg_f1}")
