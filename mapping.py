#建立氣象站與 CCTV 座標對應
import pandas as pd
import math
import os

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'cctv_list.csv')
FILTERED_CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'cctv_filtered.csv')

# 氣象署官方測站座標
STATIONS = {
    '臺中': (24.145736, 120.684075),
    '田中': (23.873803, 120.581286)
}

# 只要距離臺中或田中任何一站小於 25 公里，就視為中彰地區目標
MAX_RADIUS_KM = 25.0 

def haversine_distance(lat1, lon1, lat2, lon2):
    """計算地球球面直線距離 (公里)"""
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_mapping():
    if not os.path.exists(CSV_PATH):
        print(f" 找不到檔案: {CSV_PATH}，請確認位置 ")
        return

    df = pd.read_csv(CSV_PATH)
    
    id_col = next((col for col in df.columns if col.upper() in ['ID', 'CCTVID', 'CCTV_ID']), None)
    lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude', '緯度', 'px', 'positionlat']), None)
    lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude', 'lng', '經度', 'py', 'positionlon']), None)
    
    if not all([id_col, lat_col, lon_col]):
        print(" 在 CSV 中找不到對應的 ID 或經緯度欄位 ")
        return

    print(f" 載入原始資料共 {len(df)} 支攝影機 ")
    print(f" 啟動地理圍欄：過濾半徑 {MAX_RADIUS_KM} 公里內之設備 ")
    
    filtered_records = []
    taichung_count = 0
    tianzhong_count = 0

    
    print("CCTV_STATION_MAP = {")

    for index, row in df.iterrows():
        try:
            cctv_id = str(row[id_col]).strip()
            cam_lat = float(row[lat_col])
            cam_lon = float(row[lon_col])
        except:
            continue # 跳過座標格式異常的資料
            
        dist_taichung = haversine_distance(cam_lat, cam_lon, STATIONS['臺中'][0], STATIONS['臺中'][1])
        dist_tianzhong = haversine_distance(cam_lat, cam_lon, STATIONS['田中'][0], STATIONS['田中'][1])
        
        # 取得與兩個氣象站的最近距離
        min_dist = min(dist_taichung, dist_tianzhong)
        
        # 只要距離小於半徑 (25公里)，就保留這支監視器
        if min_dist <= MAX_RADIUS_KM:
            filtered_records.append(row) # 存入備查報表
            
            # 判斷歸屬並印出字典代碼
            if dist_taichung <= dist_tianzhong:
                closest_station = '臺中'
                dist_info = f"{dist_taichung:.1f} km"
                taichung_count += 1
            else:
                closest_station = '田中'
                dist_info = f"{dist_tianzhong:.1f} km"
                tianzhong_count += 1
                
            print(f"    '{cctv_id}': '{closest_station}',  # 距離: {dist_info}")

    print("}")
    print("\n 完成 ")
    
    if filtered_records:
        filtered_df = pd.DataFrame(filtered_records)
        filtered_df.to_csv(FILTERED_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"   鎖定 {len(filtered_df)} 支中彰地區目標 ")
        print(f"   臺中站負責 {taichung_count} 支 / 田中站負責 {tianzhong_count} 支")
        print(f"   CSV 已儲存至: {FILTERED_CSV_PATH}")
    else:
        print("  在半徑範圍內找不到目標，確認座標資料是否正確 ")

if __name__ == "__main__":
    generate_mapping()