# aiot_multimodal_aligner.py
import pandas as pd
import os
import re
from datetime import datetime, timedelta
import pytz
from pysolar.solar import get_altitude, get_azimuth

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'aiot_images_cropped')          
UV_CSV_PATH = os.path.join(DATASET_DIR, 'AIoT_UV_data.csv')
FINAL_DATASET_CSV = os.path.join(DATASET_DIR, 'aiot_multimodal_dataset.csv')

# 參數設定
MAX_TOLERANCE_MINUTES = 5 # AIoT 設備採樣頻率高，容許誤差縮小至 5 分鐘

# 彰師大寶山校區經緯度 (用於 Pysolar 計算星曆)
NCUE_LAT = 24.074
NCUE_LON = 120.560

# 台灣時區設定
tw_tz = pytz.timezone('Asia/Taipei')

def get_uv_class(uv_value):
    """將 UV 數值轉換為分類標籤 (0~4)"""
    if uv_value <= 2.9: return 0
    elif uv_value <= 5.9: return 1
    elif uv_value <= 7.9: return 2
    elif uv_value <= 10.9: return 3
    else: return 4

def align_aiot_multimodal_data():
    print(" 啟動 [AIoT 多模態特徵] 資料對齊模組...")
    
    if not os.path.exists(UV_CSV_PATH):
        print(f" 找不到 UV 數據檔: {UV_CSV_PATH}")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f" 找不到圖片資料夾: {IMAGE_DIR}")
        return

    print(" 正在載入 AIoT UV 數據...")
    
    # 防止 Excel 存檔造成的 Big5 報錯
    try:
        uv_df = pd.read_csv(UV_CSV_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        uv_df = pd.read_csv(UV_CSV_PATH, encoding='cp950')
        
    # 將時間轉換為 datetime 物件
    uv_df['Timestamp'] = pd.to_datetime(uv_df['Timestamp'])
    uv_df = uv_df[uv_df['Station'] == '彰師大'].reset_index(drop=True)
    
    if uv_df.empty:
        print(" 找不到『彰師大』的資料！請確認 csv 內容 ")
        return

    aligned_records = []
    skipped_count = 0
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" 發現 {len(image_files)} 張 AIoT 實拍圖片，開始進行時間與星曆計算...")

    for filename in image_files:
        try:
            # 使用正則表達式抓取檔名中的時間 (例如: IMG_20260516_051000.jpg)
            match = re.search(r'(\d{8}_\d{6})', filename)
            if not match:
                skipped_count += 1
                continue
                
            time_str = match.group(1)
            # 轉換為 Naive Datetime (用於時間比對)
            img_time_naive = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            # 轉換為 Aware Datetime (附帶台灣時區，Pysolar 計算必須要用)
            img_time_aware = tw_tz.localize(img_time_naive)
            
        except Exception as e:
            skipped_count += 1
            continue

        # 時間對齊 (尋找最接近的 UVI)
        time_diffs = (uv_df['Timestamp'] - img_time_naive).abs()
        closest_idx = time_diffs.idxmin()
        closest_row = uv_df.loc[closest_idx]
        closest_diff = time_diffs.min()
        
        # 檢查是否在 5 分鐘容忍度內
        if closest_diff <= timedelta(minutes=MAX_TOLERANCE_MINUTES):
            uv_val = closest_row['UV_Index']
            
            # Pysolar 天文星曆計算
            altitude = get_altitude(NCUE_LAT, NCUE_LON, img_time_aware)
            azimuth = get_azimuth(NCUE_LAT, NCUE_LON, img_time_aware)
            
            # 月份 One-Hot 編碼 (12 維)
            current_month = img_time_naive.month
            month_features = {f"month_{i}": 1 if i == current_month else 0 for i in range(1, 13)}
            
            # 整合所有特徵
            record = {
                'filename': filename,
                'mapped_station': '彰師大',
                'uv_value': uv_val,
                'uv_class': get_uv_class(uv_val),
                'time_diff_mins': round(closest_diff.total_seconds() / 60.0, 1),
                'altitude': round(altitude, 4),  # 太陽高度角
                'azimuth': round(azimuth, 4)     # 太陽方位角
            }
            record.update(month_features) # 併入 month_1 ~ month_12
            
            aligned_records.append(record)
        else:
            skipped_count += 1

    # 輸出結果
    if aligned_records:
        final_df = pd.DataFrame(aligned_records)
        final_df.to_csv(FINAL_DATASET_CSV, index=False, encoding='utf-8-sig')
        
        print(f"\n 在地化【多模態資料】對齊完成 ")
        print(f" 成功配對與計算: {len(aligned_records)} 張圖片")
        print(f" 誤差過大 (> {MAX_TOLERANCE_MINUTES} 分鐘) 或無效捨棄: {skipped_count} 張圖片")
        print(f" 最終多模態訓練集已儲存至: {FINAL_DATASET_CSV}")
        
        print("\n 最終資料預覽 (包含物理特徵) ")
        preview_cols = ['filename', 'uv_value', 'uv_class', 'altitude', 'azimuth', 'month_5']
        # 只顯示存在的預覽欄位，防止部分欄位因資料狀況缺失報錯
        preview_cols = [c for c in preview_cols if c in final_df.columns]
        print(final_df[preview_cols].head())
    else:
        print("\n 找不到任何可以對齊的資料，請檢查時間區間是否一致 ")

if __name__ == "__main__":
    align_aiot_multimodal_data()