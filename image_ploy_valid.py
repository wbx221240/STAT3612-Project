import torch
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from sklearn.preprocessing import PolynomialFeatures
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

# 定义多项式回归模型
class PolynomialModel(nn.Module):
    def __init__(self, input_dim, degree=2):
        super(PolynomialModel, self).__init__()
        self.poly = PolynomialFeatures(degree)  # 多项式转换器
        self.linear = nn.Linear(self.poly.fit_transform(np.zeros((1, input_dim))).shape[1], 1)  # 计算输出特征数

    def forward(self, x):
        # 将输入特征转换为多项式特征
        x_poly = self.poly.fit_transform(x.cpu().numpy())  # 需要在CPU上执行，以免与GPU冲突
        x_poly_tensor = torch.tensor(x_poly, dtype=torch.float32).to(device)
        return self.linear(x_poly_tensor).squeeze()

# 初始化模型、损失函数和优化器
poly_model = PolynomialModel(input_dim=1024, degree=2).to(device)  # 假设1024是输入特征的维度
loss = nn.BCEWithLogitsLoss()
trainer = torch.optim.AdamW(poly_model.parameters(), lr=0.01, weight_decay=1e-4)

# 使用验证集进行训练和评估
num_epochs = 10
for epoch in range(num_epochs):
    poly_model.train()
    # 遍历训练集（虽然这里只使用验证集，但你可以根据需要使用训练集数据）
    for features, labels in valid_loader:
        features, labels = features.to(device), labels.to(device)
        logits = poly_model(features)
        batch_loss = loss(logits, labels)
        trainer.zero_grad()
        batch_loss.backward()
        trainer.step()

    # 验证集评估
    poly_model.eval()
    valid_running_loss = 0.0
    valid_accuracy = 0.0
    valid_auc = 0.0
    valid_f1 = 0.0

    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), labels.to(device)
            logits = poly_model(features)
            probabilities = torch.sigmoid(logits)

            valid_running_loss += loss(logits, labels).item()
            valid_accuracy += accuracy(probabilities, labels)
            valid_auc += auroc_metric(probabilities, labels)
            valid_f1 += f1_metric(probabilities, labels)

    avg_loss = valid_running_loss / len(valid_loader)
    avg_accuracy = valid_accuracy / len(valid_loader)
    avg_auc = valid_auc / len(valid_loader)
    avg_f1 = valid_f1 / len(valid_loader)

    print(f"Validation Results - Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1: {avg_f1}")
