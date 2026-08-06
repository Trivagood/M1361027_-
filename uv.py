import requests
import csv
import time
import os
from datetime import datetime, timezone, timedelta
import urllib3  # 用來處理網路請求的底層套件

# 隱藏警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CSV_FILE = os.path.join(DATASET_DIR, 'train_uv.csv')

# API 設定
API_KEY = "CWA-97860E8E-3D32-4FA7-9633-5DAD3E23162C"
DATASET_ID = "O-A0003-001" 
API_URL = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{DATASET_ID}?Authorization={API_KEY}&format=JSON"

# 設定台灣時區 (UTC+8)
TW_TZ = timezone(timedelta(hours=8))
TARGET_STATIONS = ["臺中", "田中", "彰師大"]

def get_tw_now():
    """取得當下台灣時間"""
    return datetime.now(TW_TZ)

def fetch_and_save_to_local():
    tw_now = get_tw_now()
    display_time = tw_now.strftime('%H:%M:%S')
    print(f"\n[{display_time}] 正在向氣象署查詢 [中彰地區] 數據 ")

    try:
        # 加入 verify=False，繞過 SSL 憑證檢查
        response = requests.get(API_URL, timeout=10, verify=False)
        
        if response.status_code != 200:
            print(f" 連線失敗: {response.status_code}")
            return

        data = response.json()

        # 結構解析
        stations = []
        if 'records' in data:
            if 'Station' in data['records']:
                stations = data['records']['Station']
            elif 'location' in data['records']:
                stations = data['records']['location']

        records_to_save = []

        for st in stations:
            name = st.get('StationName', st.get('locationName', '未知'))

            # 篩選測站
            if TARGET_STATIONS and name not in TARGET_STATIONS:
                continue

            # 找 UVIndex
            weather_elements = st.get('WeatherElement', st.get('weatherElement', {}))
            uv_val = -99.0

            if isinstance(weather_elements, dict) and 'UVIndex' in weather_elements:
                uv_val = weather_elements['UVIndex']
            elif isinstance(weather_elements, list):
                for item in weather_elements:
                    if item['elementName'] == 'UVIndex':
                        uv_val = item['elementValue']
                        break

            # 轉浮點數並過濾無效值
            try:
                uv_float = float(uv_val)
            except:
                uv_float = -99.0

            # 只儲存大於等於 0 的有效數據
            if uv_float >= 0:
                obs_time = st.get('ObsTime', {}).get('DateTime', '')
                records_to_save.append([name, obs_time, uv_float])

        # 寫入 CSV
        if records_to_save:
            os.makedirs(DATASET_DIR, exist_ok=True)
            file_exists = os.path.exists(CSV_FILE)

            with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["Station", "Time", "UV_Index"])
                    print(f" 已建立新檔案: {CSV_FILE} ")

                writer.writerows(records_to_save)

            print(f" 已成功儲存 {len(records_to_save)} 筆中彰資料 ")
            for r in records_to_save:
                print(f"   {r[0]}: {r[2]}")
        else:
            print(" 本次無有效 UV 數據 (可能為夜間或該時段無數據更新)。")

    except Exception as e:
        print(f" 發生錯誤: {e} ")

# 主程式
if __name__ == "__main__":
    print(" 啟動 UVI 收集排程 ")
    print(" 設定觀測時段: 每日 05:00 至 18:00 ")

    while True:
        tw_now = get_tw_now()
        current_hour = tw_now.hour

        if 5 <= current_hour <= 18:
            fetch_and_save_to_local()
        else:
            display_time = tw_now.strftime('%H:%M:%S')
            print(f"\n[{display_time}] 目前非日照觀測時段，進入休眠狀態 ")

        time.sleep(600)