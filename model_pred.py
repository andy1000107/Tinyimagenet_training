import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tinyimagenet import TinyImageNet
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Step 1: 設定運算裝置 (Device)
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Step 2: 載入測試/驗證資料集 (Validation Dataset & DataLoader)
# ------------------------------------------------------------------
dataset_path = Path("~/.torchvision/tinyimagenet/").expanduser()
val_data = TinyImageNet(dataset_path, split='val', transform=None)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

print(f"Validation dataset size: {len(val_data)}")

# ------------------------------------------------------------------
# Step 3: 重建模型架構並載入訓練好的權重
# ------------------------------------------------------------------
def load_trained_model(weight_path, num_classes=200):
    # 這裡預設不用再去下載 ImageNet 官方預訓練權重，因為我們要覆蓋自己的權重
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # 載入權重，使用 map_location 確保即使在沒有 GPU 的環境也能載入
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()  # 關鍵：切換為 Evaluation 模式 (關閉 Dropout, 鎖定 BatchNorm)
    return model

weight_file = 'best_tinyimagenet_resnet18.pth'
model = load_trained_model(weight_file, num_classes=200)
print(f"Successfully loaded weights from '{weight_file}'")

# ------------------------------------------------------------------
# Step 4: 評估模型 (計算 Loss、Top-1 Acc 與 Top-5 Acc)
# ------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

running_loss = 0.0
correct_top1 = 0
correct_top5 = 0
total_samples = 0

print("\nEvaluating model on validation set...")

# with torch.no_grad() 能關閉梯度計算，大幅節省記憶體並加速測試
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 累加 Loss
        running_loss += loss.item() * inputs.size(0)

        # --- Top-1 Accuracy ---
        _, preds = torch.max(outputs, 1)
        correct_top1 += torch.sum(preds == labels.data).item()

        # --- Top-5 Accuracy ---
        # 取得機率最高的前 5 個預測類別 index
        _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
        # 檢查真實標籤是否存在於這 5 個預測結果中
        labels_reshaped = labels.view(-1, 1)
        correct_top5 += torch.sum(top5_preds == labels_reshaped).item()

        total_samples += inputs.size(0)

test_loss = running_loss / total_samples
top1_acc = (correct_top1 / total_samples) * 100
top5_acc = (correct_top5 / total_samples) * 100

print("=" * 40)
print(f" Test Loss     : {test_loss:.4f}")
print(f" Top-1 Accuracy: {top1_acc:.2f}%")
print(f" Top-5 Accuracy: {top5_acc:.2f}%")
print("=" * 40)

# ------------------------------------------------------------------
# Step 5: 視覺化抽樣預測 (Visualizing Predictions)
# ------------------------------------------------------------------
def visualize_predictions(model, dataset, num_samples=8):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(12, 6))
    model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[int(idx)]
            
            # 準備輸入 Tensor (增加 Batch 維度: [C, H, W] -> [1, C, H, W])
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred_label = torch.max(output, 1)
            
            pred_class = pred_label.item()
            
            # 畫圖
            plt.subplot(2, num_samples // 2, i + 1)
            plt.imshow(image.permute(1, 2, 0).cpu())
            
            # 判斷預測是否正確並標示顏色
            is_correct = (pred_class == true_label)
            color = 'green' if is_correct else 'red'
            
            # 嘗試取得英文標籤名稱 (若套件支援 idx_to_words)
            if hasattr(dataset, 'idx_to_words'):
                pred_name = dataset.idx_to_words[pred_class][0]
                true_name = dataset.idx_to_words[true_label][0]
                title = f"Pred: {pred_name}\nTrue: {true_name}"
            else:
                title = f"Pred: {pred_class} | True: {true_label}"
                
            plt.title(title, color=color, fontsize=9)
            plt.axis("off")
            
    plt.tight_layout()
    plt.show()

print("\nDisplaying sample predictions...")
visualize_predictions(model, val_data, num_samples=8)