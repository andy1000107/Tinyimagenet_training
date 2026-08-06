# %%
from tinyimagenet import TinyImageNet
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


val_data = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split='val', transform=None)
train_data = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split='train', transform=None)
n = len(val_data)
print(type(val_data), type(train_data))

# %%
#define data loaders
from torch.utils.data import DataLoader


train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

# %%
n_samples = 8
print(f"Showing info of {n_samples} samples...")

# 產生 n_samples 個等距 index
indices = np.linspace(0, n - 1, n_samples, dtype=int)

plt.figure(figsize=(8, 8))
for i, idx in enumerate(indices):
    image, klass = val_data[int(idx)]
    print(f"Sample {i + 1}: class {klass}")
    plt.subplot(n_samples//2, 2, i + 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

plt.show()

# %%
#Create a model using ResNet18 architecture

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=200, pretrained=True):
    # 載入預訓練權重
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    
    # 替換最後一層全連接層
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=200, pretrained=True).to(device)
print(f"Model created and moved to: {device}")

# %%
#Define the loss function and optimizer

import torch.optim as optim

# 定義交叉熵損失函數
criterion = nn.CrossEntropyLoss()

# 使用 AdamW 優化器，微調預訓練模型建議使用較小的 learning rate (例如 3e-4)
# 這裡保留 weight decay，並在 loss 中額外加入 L2 regularization，讓正則化更明確
weight_decay = 1e-2
l2_lambda = 1e-4
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=weight_decay)

# 設定學習率衰減策略
epochs = 15
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# %%
import copy
import torch


def append_training_log(log_path, train_loss, val_loss, best_acc, epoch):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open('a', encoding='utf-8') as f:
        if log_path.stat().st_size == 0:
            f.write('epoch,train_loss,val_loss,best_val_acc\n')
        f.write(f'{epoch},{train_loss:.4f},{val_loss:.4f},{best_acc:.4f}\n')


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, print_freq=20, early_stopping_patience=3, early_stopping_min_delta=1e-4):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_without_improvement = 0
    last_train_loss = None
    last_val_loss = None
    last_epoch = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # 每輪包含 Training 與 Validation 兩個階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            step_running_loss = 0.0

            for step, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 對輸入影像做 ImageNet 標準化
                mean_tensor = torch.tensor(mean, dtype=inputs.dtype, device=inputs.device).view(1, 3, 1, 1)
                std_tensor = torch.tensor(std, dtype=inputs.dtype, device=inputs.device).view(1, 3, 1, 1)
                inputs = (inputs - mean_tensor) / std_tensor

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':

                        # 額外加入 L2 regularization，防止過擬合
                        l2_reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                        loss = loss + l2_lambda * l2_reg
                        ###
                        
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    step_running_loss += loss.item()

                    if (step + 1) % print_freq == 0:
                        avg_step_loss = step_running_loss / print_freq
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"  Step [{step+1:4d}/{len(dataloader)}] - Recent Loss: {avg_step_loss:.4f} | LR: {current_lr:.6f}")
                        step_running_loss = 0.0

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_acc_val = epoch_acc.item()

            print(f"--> {phase.capitalize()} Epoch Loss: {epoch_loss:.4f} Acc: {epoch_acc_val:.4f}")

            if phase == 'train':
                last_train_loss = epoch_loss
            else:
                last_val_loss = epoch_loss
                last_epoch = epoch + 1
                append_training_log('training_log.csv', last_train_loss if last_train_loss is not None else 0.0, last_val_loss, best_acc, last_epoch)

            if phase == 'val':
                if epoch_acc_val > best_acc + early_stopping_min_delta:
                    best_acc = epoch_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_tinyimagenet_resnet18.pth')
                    epochs_without_improvement = 0
                    print(f"    [★] Saved new best model with Val Acc: {best_acc:.4f}")
                else:
                    epochs_without_improvement += 1
                    print(f"    [!] No improvement for {epochs_without_improvement} epoch(s).")

                    if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                        print(f"    [STOP] Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
                        model.load_state_dict(best_model_wts)
                        return model

    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# %%
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=epochs)


