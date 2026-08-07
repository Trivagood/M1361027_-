# aiot_中央裁切
import os
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

# 允許載入截斷的圖片，防止程式崩潰
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#  輸入 aiot 拍回來的原始照片資料夾
INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'dataset', 'images_esp32')

#  輸出：裁切完成純天空影像資料夾
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'dataset', 'esp32_images_cropped')

# 參數設定
# 預設 0.6 (60%) 避開魚眼的黑色暗角與邊緣嚴重變形區
CENTER_CROP_RATIO = 0.60  
TARGET_SIZE = (224, 224) 

def main():
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f" 找不到原始圖片資料夾: {INPUT_IMAGE_DIR}")
        print("請先建立資料夾並放入您的 AIoT 測試照片。")
        return

    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)

    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) 
                   if os.path.isfile(os.path.join(INPUT_IMAGE_DIR, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f" 總共找到 {len(image_files)} 張 AIoT 圖片準備進行中央裁切...")

    success_count = 0
    error_count = 0

    # 確保相容舊版與新版的 Pillow 語法
    resample_filter = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS) if hasattr(Image, 'Resampling') else Image.LANCZOS

    for img_name in tqdm(image_files, desc="裁切進度"):
        input_path = os.path.join(INPUT_IMAGE_DIR, img_name)
        output_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
        
        try:
            with Image.open(input_path) as raw_img:
                # 修正相機可能帶有的 EXIF 旋轉
                raw_img = ImageOps.exif_transpose(raw_img)
                img_rgb = raw_img.convert('RGB')
                
                width, height = img_rgb.size
                
                # 計算正中央裁切框
                # 決定正方形的邊長 (取圖片短邊乘以比例)
                crop_side = int(min(width, height) * CENTER_CROP_RATIO)
                
                # 計算正中心點
                cx, cy = width / 2, height / 2
                
                # 計算上下左右邊界
                left = cx - (crop_side / 2)
                top = cy - (crop_side / 2)
                right = cx + (crop_side / 2)
                bottom = cy + (crop_side / 2)
                
                # 執行正中央裁切
                cropped_img = img_rgb.crop((left, top, right, bottom))
                
                # 縮放成 224x224
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

    print("\n AIoT 影像中央裁切完成 ")
    print(f" 成功裁切: {success_count} 張")
    if error_count > 0:
        print(f" 裁切失敗/損毀: {error_count} 張")
    print(f" 裁切後的天空影像已儲存至: {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()