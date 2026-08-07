#aiot_mobilenet_遷移學習
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

# 允許載入截斷的圖片，防止 DataLoader 運作中途崩潰
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'aiot_images_cropped')         
CSV_PATH = os.path.join(DATASET_DIR, 'aiot_multimodal_dataset.csv')

# 載入第一階段 (CCTV) 訓練好的 MobileNetV2 多模態權重檔
PRETRAINED_WEIGHTS_PATH = os.path.join(BASE_DIR, 'multimodal_mobilenet_deep_best.pth')
# 本次 AIoT 微調後的最終模型儲存路徑
AIOT_SAVE_PATH = os.path.join(BASE_DIR, 'aiot_mobilenet_finetuned.pth')

# 建立多模態資料集
class AIoTMultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        # 確保索引連續
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        # 14 維物理特徵欄位
        self.phys_cols = ['altitude', 'azimuth'] + [f'month_{i}' for i in range(1, 13)]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.loc[idx, 'filename'])
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.loc[idx, 'uv_class'])
        
        # 提取物理特徵並轉換為 Tensor
        phys_feats = self.dataframe.loc[idx, self.phys_cols].values.astype('float32')
        phys_feats = torch.tensor(phys_feats)

        if self.transform:
            image = self.transform(image)

        return image, phys_feats, label

# 建立 MobileNetV2 多模態融合神經網路
class MultimodalMobileNetV2(nn.Module):
    def __init__(self, num_classes=4, phys_dim=14):
        super(MultimodalMobileNetV2, self).__init__()
        
        # 視覺特徵分支 (MobileNetV2 骨幹)
        self.mobilenet = models.mobilenet_v2(weights=None)
        num_ftrs = self.mobilenet.last_channel  # 1280 維
        self.mobilenet.classifier = nn.Identity()  # 移除原本的分類頭
        
        # 物理特徵升維網路 (14 -> 64 -> 128)
        self.phys_network = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 融合分類器 (與預訓練完全一致：(1280 + 128) -> 256 -> 4)
        fusion_dim = num_ftrs + 128
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, phys_features):
        vis_features = self.mobilenet(image)
        phys_encoded = self.phys_network(phys_features)
        
        # 晚期融合 (Late Fusion)
        fused_features = torch.cat((vis_features, phys_encoded), dim=1)
        out = self.fusion_classifier(fused_features)
        return out

# 主訓練與微調流程
def main():
    if not os.path.exists(CSV_PATH) or not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f" 找不到 CSV 檔案或預訓練權重檔，請檢查路徑設定")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 確保 uv_class 包含完整的 4 個類別 (0, 1, 2, 3)
    df['uv_class'] = df['uv_class'].astype(int).replace(4, 3)
    num_classes = 4
    
    print(f" 成功讀取 AIoT 多模態資料集，共 {len(df)} 筆數據 ")

    # 依類別比例切分訓練集與驗證集 (8:2)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['uv_class'])

    # 資料增強與標準化
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AIoTMultimodalDataset(train_df, IMAGE_DIR, transform=train_transforms)
    val_dataset = AIoTMultimodalDataset(val_df, IMAGE_DIR, transform=val_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"運算設備設定為: {device}")

    # 載入預訓練權重
    model = MultimodalMobileNetV2(num_classes=num_classes, phys_dim=14)
    print("正在載入預訓練之 MobileNetV2 多模態融合權重...")
    
    # 載入全部權重字典
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device, weights_only=False))
    model = model.to(device)
    print("權重名稱完全一致，視覺、物理分支與分類器已成功導入。")

    # 設定損失函數與分層差異化學習率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.mobilenet.parameters(), 'lr': 1e-5},       # 預訓練骨幹：微小學習率
        {'params': model.phys_network.parameters(), 'lr': 1e-3},     # 物理分支：正常學習率
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-3} # 融合分類器：正常學習率
    ])
    
    # 早停機制 (Early Stopping) 與 餘弦下降排程 (Cosine Annealing)
    num_epochs = 100     # 最大 Epochs
    patience = 10        # 早停容忍度
    no_improve_count = 0 # 連續未改善計數器

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    print(f"\n【 AIoT 深度特徵投影多模態 MobileNetV2】開始遷移學習，最大 Epochs: {num_epochs} (啟用早停機制)...")
    

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, phys_feats, labels in train_loader:
            images, phys_feats, labels = images.to(device), phys_feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, phys_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 乘上 batch_size 的方式累加總 Loss
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 每個 Epoch 結束時更新餘弦學習率排程
        scheduler.step() 

        epoch_train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # 測試階段：詳細統計與呈現各個分類猜對/猜錯的表現
        model.eval()
        val_running_loss = 0.0 # 計算驗證集總 Loss
        val_correct, val_total = 0, 0
        
        # 用於記錄 4 個類別的表現
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for images, phys_feats, labels in val_loader:
                images, phys_feats, labels = images.to(device), phys_feats.to(device), labels.to(device)
                outputs = model(images, phys_feats)
                
                loss = criterion(outputs, labels) # 計算驗證集的 Loss 軌跡
                val_running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 記錄每一筆預測
                for i in range(len(labels)):
                    real_label = labels[i].item()
                    pred_label = predicted[i].item()
                    if real_label == pred_label:
                        class_correct[real_label] += 1
                    class_total[real_label] += 1

        epoch_val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # 同時呈現 Train Loss 與 Val Loss
        print(f"\nEpoch [{epoch+1:02d}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 輸出透明且詳細的各類別猜中/猜錯細節
        print("  測試集各類別預測表現細節：")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                wrong = class_total[i] - class_correct[i]
                print(f"      - Class {i}: 總數 {class_total[i]:<4} | 猜中 {class_correct[i]:<4} | 猜錯 {wrong:<4} | 準確率 {acc:.2f}%")
            else:
                print(f"      - Class {i}: 總數 0 (本批測試集無此類別數據)")

        # 早停機制與最佳權重儲存邏輯
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0 # 連續未改善計數器
            torch.save(model.state_dict(), AIOT_SAVE_PATH)
            print(f"  已儲存最佳【 AIoT 深度特徵投影多模態 MobileNetV2】微調權重 (Val Acc: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1
            print(f"   模型未進步 ({no_improve_count}/{patience})")

        # 檢查是否觸發早停
        if no_improve_count >= patience:
            print(f"\n觸發早停機制 (Early Stopping)")
            print(f"   模型已連續 {patience} 個 Epoch 未能提升驗證準確率，判定已達邊緣端收斂極限。")
            break

    print(f"\n 【 AIoT 深度特徵投影多模態 MobileNetV2】遷移微調完成，最高測試集準確率為: {best_val_acc:.2f}%\n")

    # 量化效能評估：整合 Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    print("\n" + "="*70)
    print(" 訓練全數完成，載入最佳模型進行評估...")
    print("="*60)
    
    # 確保載入的是測試集表現最好的權重
    if os.path.exists(AIOT_SAVE_PATH):
        model.load_state_dict(torch.load(AIOT_SAVE_PATH,weights_only=False))
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