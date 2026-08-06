import os
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

# 允許載入截斷圖片，避免程式崩潰
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'dataset', 'images')
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'dataset', 'images_cropped')

# 參數設定
CROP_RATIO_TOP = 0.35  # 只保留上半部 35%
TARGET_SIZE = (224, 224) 

def main():
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f" 找不到原始圖片資料夾: {INPUT_IMAGE_DIR}")
        return

    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)

    # 確保檔案讀取
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) 
                   if os.path.isfile(os.path.join(INPUT_IMAGE_DIR, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f" 總共找到 {len(image_files)} 張 CCTV 圖片準備進行裁切...")

    success_count = 0
    error_count = 0

    # 確保相容舊版與新版的 Pillow 語法
    resample_filter = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS) if hasattr(Image, 'Resampling') else Image.LANCZOS

    for img_name in tqdm(image_files, desc="裁切進度"):
        input_path = os.path.join(INPUT_IMAGE_DIR, img_name)
        output_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
        
        try:
            with Image.open(input_path) as raw_img:
                # 修正 EXIF 旋轉，確保天空部分
                raw_img = ImageOps.exif_transpose(raw_img)
                
                # 轉為 RGB 統一格式
                img_rgb = raw_img.convert('RGB')
                width, height = img_rgb.size
                
                # 邊界計算
                left = 0
                upper = 0
                right = width
                lower = max(1, int(height * CROP_RATIO_TOP)) # 確保 lower 至少為 1
                
                # 執行裁切與縮放
                cropped_img = img_rgb.crop((left, upper, right, lower))
                resized_img = cropped_img.resize(TARGET_SIZE, resample_filter)
                
                # 高畫質存檔
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    resized_img.save(output_path, quality=95)
                else:
                    resized_img.save(output_path)
                    
                success_count += 1
                
        except Exception as e:
            # print(f" 無法處理圖片 {img_name}: {e}")
            error_count += 1

    print("\n 裁切作業完成")
    print(f"成功裁切 {success_count} 張")
    if error_count > 0:
        print(f" 裁切失敗/損毀: {error_count} 張")
    print(f" 裁切後的天空影像已儲存至: {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()