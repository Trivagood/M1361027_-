# data_aligner.py
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from pysolar.solar import get_altitude, get_azimuth

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'images_cropped')
UV_CSV_PATH = os.path.join(DATASET_DIR, 'train_uv.csv')
FINAL_DATASET_CSV = os.path.join(DATASET_DIR, 'final_dataset_multimodal.csv')

MAX_TOLERANCE_MINUTES = 10

# 監視器與氣象站的「地理對應表」 (Spatial Mapping)
CCTV_STATION_MAP = {
    'CCTV-21-0030-148-001': '臺中', 'CCTV-22-0130-057-001': '臺中', 'CCTV-22-0010-150-001': '臺中',
    'CCTV-22-0010-154-001': '臺中', 'CCTV-22-0010-156-001': '臺中', 'CCTV-22-0010-162-001': '臺中',
    'CCTV-22-0010-168-001': '臺中', 'CCTV-22-0010-173-001': '臺中', 'CCTV-22-0010-175-001': '臺中',
    'CCTV-22-0010-181-001': '臺中', 'CCTV-22-0010-183-001': '臺中', 'CCTV-22-001B-001-001': '臺中',
    'CCTV-22-001B-000-002': '臺中', 'CCTV-22-001B-015-001': '臺中', 'CCTV-22-001B-016-001': '臺中',
    'CCTV-22-001B-017-002': '臺中', 'CCTV-22-001B-018-001': '臺中', 'CCTV-22-001B-020-001': '臺中',
    'CCTV-22-001B-021-001': '臺中', 'CCTV-22-0030-150-002': '臺中', 'CCTV-22-0030-153-001': '臺中',
    'CCTV-22-0030-157-002': '臺中', 'CCTV-22-0030-158-001': '臺中', 'CCTV-22-0030-165-001': '臺中',
    'CCTV-22-0030-166-001': '臺中', 'CCTV-22-0030-168-001': '臺中', 'CCTV-22-0030-171-001': '臺中',
    'CCTV-22-0030-175-002': '臺中', 'CCTV-22-0030-177-001': '臺中', 'CCTV-22-0030-187-001': '臺中',
    'CCTV-22-0030-189-001': '臺中', 'CCTV-22-0030-189-002': '臺中', 'CCTV-22-0030-193-001': '臺中',
    'CCTV-22-0030-195-001': '臺中', 'CCTV-22-0030-199-001': '臺中', 'CCTV-22-0030-199-002': '臺中',
    'CCTV-22-0100-007-002': '臺中', 'CCTV-22-0100-017-001': '臺中', 'CCTV-22-0100-019-001': '臺中',
    'CCTV-22-0100-020-002': '臺中', 'CCTV-22-010B-004-002': '臺中', 'CCTV-22-0120-004-001': '臺中',
    'CCTV-22-0120-007-001': '臺中', 'CCTV-22-0120-011-001': '臺中', 'CCTV-22-0130-064-001': '臺中',
    'CCTV-22-0130-065-001': '臺中', 'CCTV-22-0170-000-001': '臺中', 'CCTV-22-0170-000-002': '臺中',
    'CCTV-22-0170-007-002': '臺中', 'CCTV-22-0170-007-001': '臺中', 'CCTV-22-0170-015-001': '臺中',
    'CCTV-22-0610-152-001': '臺中', 'CCTV-23-0010-187-001': '臺中', 'CCTV-23-0010-188-001': '臺中',
    'CCTV-23-0010-195-001': '臺中', 'CCTV-23-0010-199-001': '田中', 'CCTV-23-0010-214-002': '田中',
    'CCTV-23-0010-216-001': '田中', 'CCTV-23-001C-002-003': '臺中', 'CCTV-23-001C-002-004': '臺中',
    'CCTV-23-0140-004-001': '臺中', 'CCTV-23-014C-001-001': '臺中', 'CCTV-23-0170-020-001': '臺中',
    'CCTV-23-0170-028-001': '臺中', 'CCTV-23-0170-040-001': '田中', 'CCTV-23-0190-002-004': '臺中',
    'CCTV-23-0190-002-002': '臺中', 'CCTV-23-0190-002-003': '臺中', 'CCTV-23-0190-005-001': '田中',
    'CCTV-28-074A-000-001': '臺中', 'CCTV-23-074A-004-001': '臺中', 'CCTV-23-074A-009-001': '田中',
    'CCTV-23-1440-003-001': '田中', 'CCTV-23-1440-003-002': '田中', 'CCTV-23-1440-007-001': '田中',
    'CCTV-23-1440-007-002': '田中', 'CCTV-23-1480-016-001': '田中', 'CCTV-23-1480-017-001': '田中',
    'CCTV-24-0030-202-001': '田中', 'CCTV-24-0030-203-001': '田中', 'CCTV-24-0030-207-001': '田中',
    'CCTV-24-0030-209-001': '田中', 'CCTV-24-0030-209-002': '田中', 'CCTV-24-0030-219-001': '田中',
    'CCTV-24-0030-220-001': '田中', 'CCTV-24-0030-222-001': '田中', 'CCTV-24-0030-223-002': '田中',
    'CCTV-24-0030-225-001': '田中', 'CCTV-24-0030-227-001': '田中', 'CCTV-24-0030-230-001': '田中',
    'CCTV-24-0140-015-001': '臺中', 'CCTV-24-0140-016-001': '田中', 'CCTV-24-0140-019-001': '田中',
    'CCTV-24-0140-022-001': '田中', 'CCTV-24-0140-023-001': '田中', 'CCTV-24-0140-024-001': '臺中',
    'CCTV-24-014B-007-001': '田中', 'CCTV-24-0160-000-001': '田中', 'CCTV-25-0080-000-001': '臺中',
    'CCTV-25-0080-001-001': '臺中', 'CCTV-25-0080-005-001': '臺中', 'CCTV-25-0080-013-003': '臺中',
    'CCTV-25-0080-013-002': '臺中', 'CCTV-25-0080-015-001': '臺中', 'CCTV-25-0210-000-001': '臺中',
    'CCTV-25-0210-017-001': '臺中', 'CCTV-24-0140-032-001': '臺中', 'CCTV-24-0140-040-001': '臺中',
    'CCTV-26-0210-025-001': '臺中', 'CCTV-24-0160-008-001': '田中', 'CCTV-24-0160-009-001': '田中',
    'CCTV-28-0630-001-001': '臺中', 'CCTV-28-0630-002-001': '臺中', 'CCTV-28-0630-004-001': '臺中',
    'CCTV-28-0630-006-002': '臺中', 'CCTV-28-0630-011-001': '臺中', 'CCTV-28-0630-015-001': '臺中',
    'CCTV-28-0630-017-001': '田中', 'CCTV-28-0740-001-001': '臺中', 'CCTV-28-0740-005-001': '臺中',
    'CCTV-28-0740-010-003': '臺中', 'CCTV-28-0760-013-001': '田中', 'CCTV-28-0760-013-002': '田中',
    'CCTV-28-0760-017-001': '田中', 'CCTV-28-0760-017-002': '田中', 'CCTV-23-0760-019-001': '田中',
    'CCTV-23-0760-022-001': '田中', 'CCTV-23-0760-022-002': '田中', 'CCTV-23-0760-023-001': '田中',
    'CCTV-21-0030-148-002': '臺中', 'CCTV-23-0170-021-001': '臺中', 'CCTV-22-0010-185-001': '臺中',
    'CCTV-22-0030-164-001': '臺中', 'CCTV-22-0030-200-001': '臺中', 'CCTV-24-014B-014-001': '田中',
    'CCTV-24-0140-039-001': '臺中', 'CCTV-23-0140-009-001': '臺中', 'CCTV-23-0140-009-002': '臺中',
    'CCTV-23-0010-207-002': '田中', 'CCTV-23-0010-221-002': '田中', 'CCTV-22-0610-160-002': '臺中',
    'CCTV-22-0610-145-001': '臺中', 'CCTV-24-0140-038-001': '臺中', 'CCTV-24-0140-014-001': '臺中',
    'CCTV-28-0630-014-001': '臺中', 'CCTV-23-0010-204-001': '田中', 'CCTV-23-0010-207-003': '田中',
    'CCTV-22-0030-195-002': '臺中', 'CCTV-24-0030-211-001': '田中', 'CCTV-22-0610-147-001': '臺中',
    'CCTV-22-0610-150-001': '臺中', 'CCTV-22-0610-154-001': '臺中', 'CCTV-22-0610-155-001': '臺中',
    'CCTV-22-0610-157-001': '臺中', 'CCTV-23-0610-164-001': '臺中', 'CCTV-23-0610-169-001': '臺中',
    'CCTV-23-0610-187-001': '田中', 'CCTV-23-0610-189-001': '田中', 'CCTV-23-0610-166-001': '臺中',
    'CCTV-23-0610-166-002': '臺中', 'CCTV-23-061B-000-001': '臺中', 'CCTV-23-0610-169-002': '臺中',
    'CCTV-23-0610-169-003': '臺中', 'CCTV-24-0030-224-001': '田中', 'CCTV-22-0610-148-001': '臺中',
    'CCTV-22-0610-155-002': '臺中', 'CCTV-23-0610-163-001': '臺中', 'CCTV-23-0610-168-001': '臺中',
    'CCTV-23-0010-205-001': '田中', 'CCTV-23-0010-206-001': '田中', 'CCTV-24-014B-003-001': '田中',
    'CCTV-24-0160-005-003': '田中', 'CCTV-24-0160-006-001': '田中', 'CCTV-24-0160-006-002': '田中',
    'CCTV-24-0160-005-002': '田中', 'CCTV-22-0610-151-004': '臺中', 'CCTV-24-0030-224-002': '田中',
    'CCTV-24-0030-216-001': '田中', 'CCTV-24-0030-227-002': '田中', 'CCTV-23-0170-020-002': '臺中',
    'CCTV-23-0170-022-001': '臺中', 'CCTV-23-061B-001-002': '臺中', 'CCTV-22-0610-157-003': '臺中',
    'CCTV-22-0170-017-001': '臺中', 'CCTV-24-0140-031-002': '臺中', 'CCTV-51-0010-228-002': '田中',
    'CCTV-51-0010-230-001': '田中', 'CCTV-51-0010-236-002': '田中', 'CCTV-51-0010-240-001': '田中',
    'CCTV-51-0010-241-002': '田中', 'CCTV-51-0010-242-001': '田中', 'CCTV-51-001D-004-001': '田中',
    'CCTV-51-001D-006-001': '田中', 'CCTV-51-001D-006-002': '田中', 'CCTV-51-001D-011-001': '田中',
    'CCTV-51-001D-012-001': '田中', 'CCTV-51-0030-237-002': '田中', 'CCTV-51-0030-239-002': '田中',
    'CCTV-51-0030-242-001': '田中', 'CCTV-51-0030-245-001': '田中', 'CCTV-51-0030-247-001': '田中',
    'CCTV-51-0030-256-003': '田中', 'CCTV-51-0030-257-003': '田中', 'CCTV-22-0030-166-002': '臺中',
    'CCTV-23-0010-191-001': '臺中', 'CCTV-28-0740-034-001': '臺中', 'CCTV-28-0740-035-001': '臺中',
    'CCTV-22-1180-004-001': '臺中', 'CCTV-22-0030-199-003': '臺中', 'CCTV-22-0030-199-004': '臺中',
    'CCTV-24-0160-011-001': '田中', 'CCTV-28-0740-013-001': '臺中', 'CCTV-28-0740-015-001': '臺中',
    'CCTV-28-0740-017-001': '臺中', 'CCTV-28-0740-018-001': '臺中', 'CCTV-28-0740-019-001': '臺中',
    'CCTV-28-0740-019-002': '臺中', 'CCTV-28-0740-016-001': '臺中', 'CCTV-51-0780-036-001': '田中',
    'CCTV-51-0780-039-001': '田中', 'CCTV-51-0780-042-001': '田中', 'CCTV-28-0740-000-001': '臺中',
    'CCTV-28-0740-001-002': '臺中', 'CCTV-28-0740-004-001': '臺中', 'CCTV-28-0740-005-003': '臺中',
    'CCTV-28-0740-006-001': '臺中', 'CCTV-28-0740-008-001': '臺中', 'CCTV-28-0740-009-001': '臺中',
    'CCTV-28-0740-010-002': '臺中', 'CCTV-28-0740-011-001': '臺中', 'CCTV-28-0740-012-001': '臺中',
    'CCTV-28-0740-013-002': '臺中', 'CCTV-28-0740-013-003': '臺中', 'CCTV-28-0740-014-001': '臺中',
    'CCTV-28-0740-014-002': '臺中', 'CCTV-28-0740-014-003': '臺中', 'CCTV-28-0740-015-003': '臺中',
    'CCTV-28-0740-017-002': '臺中', 'CCTV-28-0740-017-003': '臺中', 'CCTV-28-0740-019-003': '臺中',
    'CCTV-28-0740-021-002': '臺中', 'CCTV-28-0740-024-001': '臺中', 'CCTV-28-0740-025-001': '臺中',
    'CCTV-28-0740-027-001': '臺中', 'CCTV-28-0740-030-002': '臺中', 'CCTV-28-0740-032-001': '臺中',
    'CCTV-28-0740-038-001': '臺中', 'CCTV-28-074A-000-002': '臺中', 'CCTV-28-0740-007-002': '臺中',
    'CCTV-28-0740-003-001': '臺中', 'CCTV-28-0740-033-001': '臺中', 'CCTV-28-0740-034-002': '臺中',
    'CCTV-28-0740-002-001': '臺中', 'CCTV-28-0740-016-002': '臺中', 'CCTV-28-0740-022-001': '臺中',
    'CCTV-28-0740-029-001': '臺中', 'CCTV-28-0740-035-002': '臺中', 'CCTV-28-0740-036-003': '臺中',
    'CCTV-28-0740-036-002': '臺中', 'CCTV-28-0740-038-002': '臺中', 'CCTV-28-074A-001-002': '臺中',
    'CCTV-22-0120-008-001': '臺中', 'CCTV-28-0760-003-001': '田中', 'CCTV-28-0760-004-001': '田中',
    'CCTV-28-0760-006-001': '田中', 'CCTV-28-0760-007-001': '田中', 'CCTV-28-0760-009-001': '田中',
    'CCTV-28-0760-010-001': '田中'
}

DEFAULT_STATION = '臺中'

# 內建兩個測站的經緯度，供 pysolar 計算使用
STATION_COORDS = {
    '臺中': (24.1456, 120.6840),
    '田中': (23.8582, 120.5824)
}

def get_uv_class(uv_value):
    """將 UV 數值轉換為分類標籤 (0~4)"""
    if uv_value <= 2.9: return 0
    elif uv_value <= 5.9: return 1
    elif uv_value <= 7.9: return 2
    elif uv_value <= 10.9: return 3
    else: return 4

def align_data():
    print(" 啟動資料對齊模組 ")
    
    if not os.path.exists(UV_CSV_PATH) or not os.path.exists(IMAGE_DIR):
        print(" 找不到數據檔或裁切後的圖片資料夾 ")
        return

    print(" 正在載入 UV 數據 ")
    uv_df = pd.read_csv(UV_CSV_PATH)
    
    # 處理 CSV 時間的時區問題
    uv_df['Time'] = pd.to_datetime(uv_df['Time'])
    if uv_df['Time'].dt.tz is not None:
        uv_df['Time'] = uv_df['Time'].dt.tz_localize(None)
    
    aligned_records = []
    skipped_count = 0
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" 找到 {len(image_files)} 張天空圖片，開始進行比對與星曆計算...")
    
    # 設定台灣時區
    tw_tz = pytz.timezone('Asia/Taipei')

    for filename in image_files:
        try:
            # 倒數索引 (Negative Indexing) 進行解析
            # 無論前面有沒有 img_ 或是多餘的底線，都能精準抓到最後面的三個參數
            basename = filename.rsplit('.', 1)[0] # 去除 .jpg 副檔名
            parts = basename.split('_')
            
            cctv_id = parts[-1]   # 倒數第 1 個必定是 CCTV ID
            time_str = parts[-2]  # 倒數第 2 個必定是時間 (HHMMSS)
            date_str = parts[-3]  # 倒數第 3 個必定是日期 (YYYYMMDD)
            
            # 解析出 datetime 並加上台灣時區
            img_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            img_time_aware = tw_tz.localize(img_time)
            
        except Exception as e:
            # print(f" 檔名無法解析，跳過: {filename}, 錯誤: {e}")
            skipped_count += 1
            continue
            
        # 空間對齊
        target_station = CCTV_STATION_MAP.get(cctv_id, DEFAULT_STATION)
        station_uv_df = uv_df[uv_df['Station'] == target_station]
        
        if station_uv_df.empty:
            skipped_count += 1
            continue

        # 時間對齊
        time_diffs = (station_uv_df['Time'] - img_time).abs()
        closest_idx = time_diffs.idxmin()
        closest_row = station_uv_df.loc[closest_idx]
        closest_diff = time_diffs.min()
        
        if closest_diff <= timedelta(minutes=MAX_TOLERANCE_MINUTES):
            uv_val = closest_row['UV_Index']
            
            # 物理特徵工程 (Feature Engineering)
            # 取得測站經緯度
            lat, lon = STATION_COORDS[target_station]
            
            # 計算太陽高度角與方位角 (使用 pysolar)
            altitude = get_altitude(lat, lon, img_time_aware)
            azimuth = get_azimuth(lat, lon, img_time_aware)
            
            # 月份 One-Hot Encoding (產生 month_1 到 month_12 共 12 個欄位)
            current_month = img_time.month
            month_features = {f"month_{i}": 1 if i == current_month else 0 for i in range(1, 13)}
            
            # 將所有特徵整合成一筆記錄
            record = {
                'filename': filename,
                'cctv_id': cctv_id,
                'mapped_station': target_station,
                'uv_value': uv_val,
                'uv_class': get_uv_class(uv_val),
                'time_diff_mins': round(closest_diff.total_seconds() / 60.0, 1),
                'altitude': round(altitude, 4),  # 太陽高度角
                'azimuth': round(azimuth, 4)     # 太陽方位角
            }
            # 併入月份的 12 個欄位
            record.update(month_features)
            
            aligned_records.append(record)
        else:
            skipped_count += 1

    if aligned_records:
        final_df = pd.DataFrame(aligned_records)
        final_df.to_csv(FINAL_DATASET_CSV, index=False, encoding='utf-8-sig')
        
        print(f"\n 對齊與特徵工程已完成 ")
        print(f" 成功處理: {len(aligned_records)} 張圖片 ")
        print(f" 誤差過大或無法解析: {skipped_count} 張圖片 ")
        print(f" 最終訓練資料表已儲存至: {FINAL_DATASET_CSV}")
        print("\n--- 預覽新增的物理特徵 ---")
        # 顯示重點物理特徵欄位
        preview_cols = ['filename', 'altitude', 'azimuth', 'month_1', 'month_2','month_3','month_4','month_5', 'month_12']
        print(final_df[preview_cols].head())
    else:
        print("\n 找不到任何可以對齊的資料。")

if __name__ == "__main__":
    align_data()