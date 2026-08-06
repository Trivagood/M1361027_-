# cctv_collector.py
import pandas as pd
import requests
import os
import time
from datetime import datetime, timezone, timedelta
import urllib3  # 處理底層網路請求

# 隱藏警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CSV_PATH = os.path.join(DATASET_DIR, 'cctv_list.csv')
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')

os.makedirs(IMAGE_DIR, exist_ok=True)

# 監視器 ID
TARGET_CCTV_IDS = [
    'CCTV-28-0740-002-001',
    'CCTV-22-0030-165-001',
    'CCTV-24-0030-207-001',
    'CCTV-23-1440-001-003',
    'CCTV_28_0740_019_003',
    'CCTV-28-0630-011-001',
    'CCTV-28-0740-038-002',
    'CCTV-28-0740-035-002',
    'CCTV-23-0610-205-001',
    'CCTV-24-0140-024-001',
    'CCTV-23-0610-193-003',
    'CCTV-21-0610-130-003',
    'CCTV-22-0610-134-001',
    'CCTV-23-1440-001-002',
    'CCTV-28-0630-011-001',
    'CCTV-28-0740-016-001',
    'CCTV-28-0740-004-001',
    'CCTV-23-0610-210-001',
    'CCTV-23-0610-205-001'
]

# 設定台灣時區 (UTC+8)
TW_TZ = timezone(timedelta(hours=8))

def get_tw_now():
    """取得當下台灣時間"""
    return datetime.now(TW_TZ)

def download_target_images():
    if not os.path.exists(CSV_PATH):
        print(f" 找不到 CCTV 名冊: {CSV_PATH}，請確認檔案位置 ")
        return

    df = pd.read_csv(CSV_PATH)
    target_df = df[df['ID'].isin(TARGET_CCTV_IDS)]
    
    if target_df.empty:
        print(" 找不到目標 ID，請檢查 TARGET_CCTV_IDS 設定。")
        return
        
    tw_now = get_tw_now()
    current_time_str = tw_now.strftime("%Y%m%d_%H%M%S")
    display_time = tw_now.strftime('%H:%M:%S')
    
    print(f"\n[{display_time}] 開始下載 {len(target_df)} 支目標攝影機 ")
    
    success_count = 0
    for index, row in target_df.iterrows():
        cctv_id = str(row['ID'])
        url = row['URL']
        
        filename = f"img_{current_time_str}_{cctv_id}.jpg"
        save_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            # 加入 verify=False，繞過 SSL 檢查
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', '').lower():
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                if os.path.getsize(save_path) > 2048:
                    success_count += 1
                    print(f" 成功: {filename}")
                else:
                    os.remove(save_path)
            else:
                print(f" 失敗 ({cctv_id}): 狀態碼 {response.status_code} 或無效圖片")
        except Exception as e:
            print(f" 連線錯誤 ({cctv_id}): {e}")
            
        time.sleep(1)

    print(f" 本輪結束: 成功下載 {success_count} / {len(target_df)} 張圖片。")

# --- 主程式 ---
if __name__ == "__main__":
    print(" 啟動本機端 CCTV 影像收集排程 ")
    print("設定觀測時段: 每日 05:00 至 18:00")
    
    while True:
        # 取得正確的台灣時間
        tw_now = get_tw_now()
        current_hour = tw_now.hour

        # 判斷是否為日照時段
        if 5 <= current_hour <= 18:
            download_target_images()
        else:
            display_time = tw_now.strftime('%H:%M:%S')
            print(f"\n[{display_time}] 目前非日照觀測時段，進入休眠狀態...")
            
        # 3. 執行完畢後，再去睡覺 (600 秒 = 10 分鐘)
        time.sleep(600)