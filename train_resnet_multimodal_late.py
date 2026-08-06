# train_resnet_multimodal.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

#允許載入截斷的圖片，防止 DataLoader 運作中途崩潰
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. 絕對路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'images_cropped')
CSV_PATH = os.path.join(DATASET_DIR, 'final_dataset_multimodal.csv')

class UVMultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.phys_cols = ['altitude', 'azimuth'] + [f'month_{i}' for i in range(1, 13)]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.loc[idx, 'filename'])
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        phys_features = self.dataframe.loc[idx, self.phys_cols].values.astype(float)
        phys_features = torch.tensor(phys_features, dtype=torch.float32)
        label = int(self.dataframe.loc[idx, 'uv_class'])

        return image, phys_features, label


# 深度特徵投影多模態網路 (Deep Projection Multimodal Fusion)
class MultimodalResNet18(nn.Module):
    def __init__(self, num_classes=4, phys_dim=14):
        super(MultimodalResNet18, self).__init__()
        
        # 視覺特徵分支
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # 物理特徵升維網路
        self.phys_network = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 融合分類器 (Fusion Classifier)
        fusion_dim = num_ftrs + 128
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, phys_features):
        vis_features = self.resnet(image) 
        phys_encoded = self.phys_network(phys_features) 
        
        fused_features = torch.cat((vis_features, phys_encoded), dim=1) 
        out = self.fusion_classifier(fused_features)
        return out

def main():
    if not os.path.exists(CSV_PATH):
        print(f" 找不到 {CSV_PATH} ")
        return

    df = pd.read_csv(CSV_PATH)
    print(f" 原始資料總共讀取到 {len(df)} 筆。\n")

    print(" 執行特徵正規化 ")
    df['altitude'] = df['altitude'] / 90.0
    df['azimuth'] = df['azimuth'] / 360.0
    print(" 太陽高度角與方位角已縮放至安全數值範圍。\n")

    df['uv_class'] = df['uv_class'].astype(int).replace(4, 3)
    num_classes = 4

    print(" 執行資料平衡 (Undersampling) ")
    class0_df = df[df['uv_class'] == 0]
    class0_original_count = len(class0_df)
    
    if class0_original_count > 4600:
        class0_df = class0_df.sample(n=4600, random_state=42) 
        
    other_classes_df = df[df['uv_class'] != 0]
    df = pd.concat([class0_df, other_classes_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f" 經過類別合併與欠採樣後，有效訓練資料剩餘 {len(df)} 筆。\n")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['uv_class'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"   -> 訓練集: {len(train_df)} 筆, 驗證集: {len(val_df)} 筆")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = UVMultimodalDataset(train_df, IMAGE_DIR, transform=train_transforms)
    val_dataset = UVMultimodalDataset(val_df, IMAGE_DIR, transform=val_transforms)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 使用運算設備: {device}")
    model = MultimodalResNet18(num_classes=num_classes, phys_dim=14).to(device)

    weights = []
    train_total = len(train_df)
    for i in range(num_classes):
        count = len(train_df[train_df['uv_class'] == i])
        weights.append(train_total / (num_classes * count) if count > 0 else 0.0)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam([
        {'params': model.resnet.parameters(), 'lr': 1e-5},
        {'params': model.phys_network.parameters(), 'lr': 1e-3}, # 物理分支給予較大 LR           
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-3} 
    ])
    
    # 早停機制 (Early Stopping) 與 學習率排程
    num_epochs = 100     # 最大容許 Epochs
    patience = 10        # 早停容忍度
    no_improve_count = 0 # 連續未進步計數器

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"\n 開始訓練【深度特徵投影多模態 ResNet18】，最大 Epochs: {num_epochs} (啟用早停機制) ")
    
    best_val_acc = 0.0
    model_save_path = os.path.join(BASE_DIR, 'multimodal_resnet18_deep_best.pth') 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, phys_feats, labels in train_loader:
            images = images.to(device)
            phys_feats = phys_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, phys_feats)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step() # 更新學習率

        epoch_train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # 測試階段
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for images, phys_feats, labels in val_loader:
                images = images.to(device)
                phys_feats = phys_feats.to(device)
                labels = labels.to(device)
                
                outputs = model(images, phys_feats)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    real_label = labels[i].item()
                    pred_label = predicted[i].item()
                    if real_label == pred_label:
                        class_correct[real_label] += 1
                    class_total[real_label] += 1

        val_acc = 100 * val_correct / val_total
        print(f"\nEpoch [{epoch+1:02d}/{num_epochs}] Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        print(" 測試集各類別預測細節：")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                wrong = class_total[i] - class_correct[i]
                print(f"      - Class {i}: 總數 {class_total[i]:<4} | 猜中 {class_correct[i]:<4} | 猜錯 {wrong:<4} | 準確率 {acc:.1f}%")
            else:
                print(f"      - Class {i}: 總數 0 (本批測試集無此類別)")

        # 早停機制邏輯判斷
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0 # 計數器歸零
            torch.save(model.state_dict(), model_save_path)
            print(f" 已儲存目前最佳【深度特徵投影多模態 ResNet18】(準確率: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1
            print(f" 模型未進步 ({no_improve_count}/{patience})")

        # 觸發早停
        if no_improve_count >= patience:
            print(f"\n 觸發早停機制 (Early Stopping)")
            print(f"   模型已連續 {patience} 個 Epoch 未能提升測試準確率，判定已達收斂極限 ")
            break

    print(f"\n【深度特徵投影多模態 ResNet18】最佳準確率為: {best_val_acc:.2f}%")

    # 量化效能評估：整合 Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    print("\n" + "="*70)
    print(" 訓練完成，載入最佳模型進行評估...")
    print("="*60)
    
    # 確保載入的是測試集表現最好的權重
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path,weights_only=False))
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    # 盲測收集預測結果
    with torch.no_grad():
        for images, phys_feats, labels in val_loader:
            images, phys_feats = images.to(device), phys_feats.to(device)
            outputs = model(images, phys_feats)
            _, predicted = torch.max(outputs.data, 1)
            
            #將變數名稱對齊上方宣告的 labels
            all_true_labels.extend(labels.numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    # 同步確認下方這兩行轉成 numpy array 的變數名稱是否有對齊：
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)

    # 整合四大指標
    # 指標一：整體準確率 (Accuracy)
    final_accuracy = accuracy_score(y_true, y_pred)

    # 指標二 & 三：精確率 (Precision)、召回率 (Recall)、F1-score (綜合指標)
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes)
    )

    # 指標四：標準混淆矩陣 (Confusion Matrix)
    conf_mat = confusion_matrix(y_true, y_pred)

    # 輸出報表
    print(f"\n 【量化指標一：整體全域評估】")
    print(f"    驗證集整體準確率 (Overall Accuracy): {final_accuracy * 100:.2f} %")

    print(f"\n 【量化指標二 & 三：類別精細適應性指標】")
    print(f"   " + "-"*65)
    print(f"   風險等級   |  精確率 (Precision)  |  召回率 (Recall)  |  F1-Score (綜合指標)")
    print(f"   " + "-"*65)
    for i in range(num_classes):
        print(f"   Class {i:<4} |       {precisions[i]*100:6.2f} %       |     {recalls[i]*100:6.2f} %     |       {f1_scores[i]:.4f}")
    print(f"   " + "-"*65)

    print(f"\n【四、決策模糊分析：混淆矩陣 (Confusion Matrix)】")
    print(f"   說明：橫軸(Columns)為模型預測值；縱軸(Rows)為真實觀測值\n")
    
    # 轉換為 DataFrame 表格排版輸出
    cm_df = pd.DataFrame(conf_mat, 
                         index=[f'True_Class_{i}' for i in range(num_classes)], 
                         columns=[f'Pred_Class_{i}' for i in range(num_classes)])
    print(cm_df)
    print("\n" + "="*60)
    print("評估指標報表導出完畢")
    print("="*60)

if __name__ == "__main__":
    main()