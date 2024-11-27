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
# 初始化指标
accuracy = Accuracy(task="binary")
auroc_metric = AUROC(task="binary")
f1_metric = F1Score(task="binary")

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
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.LeakyReLU(),
    nn.Linear(128, 1)
)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

# K 折交叉验证
k_folds = 5
num_epochs = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
dataset = ReadmissionDataset(matched_files)

for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.image_paths, dataset.labels)):
    print(f"Fold {fold + 1}/{k_folds}")

    # 获取训练和验证子集
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

    # 初始化模型、损失函数和优化器
    image_net.apply(init_weights)
    loss = nn.BCEWithLogitsLoss()
    trainer = torch.optim.AdamW(image_net.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练模型
    for epoch in range(num_epochs):
        image_net.train()
        for features, labels in train_loader:
            logits = image_net(features).squeeze()
            batch_loss = loss(logits, labels)
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
            logits = image_net(features).squeeze()
            probabilities = torch.sigmoid(logits)
            
            valid_running_loss += loss(logits, labels).item()
            valid_accuracy += accuracy(probabilities, labels)
            valid_auc += auroc_metric(probabilities, labels)
            valid_f1 += f1_metric(probabilities, labels)

    avg_loss = valid_running_loss / len(valid_loader)
    avg_accuracy = valid_accuracy / len(valid_loader)
    avg_auc = valid_auc / len(valid_loader)
    avg_f1 = valid_f1 / len(valid_loader)

    print(f"Validation Results for Fold {fold + 1} - Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1: {avg_f1}")
