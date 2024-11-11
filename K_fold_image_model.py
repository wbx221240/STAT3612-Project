import torch
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from d2l import torch as d2l
from torchmetrics import Accuracy, AUROC, F1Score

# 初始化指标
accuracy = Accuracy(task="binary")
auroc_metric = AUROC(task="binary")
f1_metric = F1Score(task="binary")

# 读取数据
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
    full_path = os.path.join('C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features', path)
    if path in folder_images:
        matched_files[full_path] = label

# 加载图像特征
def load_image_features(image_filename):
    with open(image_filename, 'rb') as f:
        image_features = pickle.load(f) 
    return torch.tensor(image_features, dtype=torch.float32)

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
        image_features = load_image_features(image_path)
        return image_features, torch.tensor(label, dtype=torch.float32)

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

# K 折交叉验证设置
k_folds = 5
num_epochs = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
dataset = ReadmissionDataset(matched_files)

# 存储每一折的结果
fold_results = {'accuracy': [], 'auc': [], 'f1': []}

# 开始 K 折交叉验证
for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.image_paths, dataset.labels)):
    print(f"Fold {fold+1}/{k_folds}")
    
    # 获取训练和验证子集
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    image_net.apply(init_weights)  # 重新初始化权重
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
    
    # 验证模型
    image_net.eval()
    val_running_loss = 0.0
    total_accuracy = 0.0
    total_auc = 0.0
    total_f1 = 0.0
    
    with torch.no_grad():
        for features, labels in val_loader:
            logits = image_net(features).squeeze()
            probabilities = torch.sigmoid(logits)
            
            val_running_loss += loss(logits, labels).item()
            batch_accuracy = accuracy(probabilities, labels)
            batch_auc = auroc_metric(probabilities, labels)
            batch_f1 = f1_metric(probabilities, labels)
            
            total_accuracy += batch_accuracy
            total_auc += batch_auc
            total_f1 += batch_f1
    
    # 计算每折的平均分数
    avg_loss = val_running_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    avg_auc = total_auc / len(val_loader)
    avg_f1 = total_f1 / len(val_loader)
    
    print(f"Fold {fold+1} - Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1: {avg_f1}")
    
    # 保存每一折的结果
    fold_results['accuracy'].append(avg_accuracy)
    fold_results['auc'].append(avg_auc)
    fold_results['f1'].append(avg_f1)

# 输出 K 折的平均分数
print("\nCross-Validation Results:")
print(f"Average Accuracy: {np.mean(fold_results['accuracy'])}")
print(f"Average AUC: {np.mean(fold_results['auc'])}")
print(f"Average F1: {np.mean(fold_results['f1'])}")
