import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from ifisheye import *
from utils import *

def convert_one_image_circular(src_image_path, des_image_dir, distortion_coefficient=0.6):
    image_name = os.path.splitext(os.path.basename(src_image_path))[0]
    img = cv2.imread(src_image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    target_dim = max(h, w)
    
    # 1. Resize ảnh gốc thành hình vuông
    img_square = cv2.resize(img, (target_dim, target_dim))

    # 2. Tạo hiệu ứng Fisheye
    warped_img = fish(img_square, distortion_coefficient)

    # -------------------------------------------------------------
    # FIX LỖI KÊNH MÀU Ở ĐÂY:
    # Hàm fish() trả về ảnh 4 kênh (RGBA), ta cắt lấy 3 kênh (BGR) 
    # để khớp với mặt nạ mask bên dưới.
    if warped_img.shape[2] == 4:
        warped_img = warped_img[:, :, :3]
    # -------------------------------------------------------------

    # 3. THUẬT TOÁN TỰ ĐỘNG DÒ TÌM VÀ CẮT BỎ SẠCH SẼ VIỀN ĐEN
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Lấy khung chữ nhật ôm sát nhất phần ảnh có màu
        c = max(contours, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(c)
        cropped_img = warped_img[y:y+h_c, x:x+w_c]
    else:
        cropped_img = warped_img

    # 4. KÉO GIÃN ẢNH ĐÃ CẮT CHO TRÀN ĐẦY KHUNG VUÔNG LỚN
    stretched_img = cv2.resize(cropped_img, (target_dim, target_dim))

    # 5. ÁP DỤNG MẶT NẠ HÌNH TRÒN ĐỂ TẠO RA ẢNH TRÒN XOE
    mask = np.zeros((target_dim, target_dim, 3), dtype=np.uint8)
    center = (target_dim // 2, target_dim // 2)
    radius = (target_dim // 2) - 2 # Trừ đi 2 pixel để viền tròn được mượt
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Lệnh chập ảnh sẽ chạy mượt mà vì cả 2 đều là 3 kênh màu
    final_img = cv2.bitwise_and(stretched_img, mask)

    # Lưu ảnh
    img_path = os.path.join(des_image_dir, f"{image_name}.jpg")
    cv2.imwrite(img_path, final_img)


def convert_all_images_circular(src_image_paths, des_image_dir, distortion_coefficient=0.6):
    print(f"[INFO]: Đang xử lý {len(src_image_paths)} frames ảnh...")
    for img_path in tqdm(src_image_paths):
        convert_one_image_circular(img_path, des_image_dir, distortion_coefficient)
    print("[INFO] Hoàn thành!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo hiệu ứng Fisheye TRÒN VÀ CONG")
    parser.add_argument("--src_path", type=str, required=True, help="Đường dẫn đến frames gốc")
    parser.add_argument("--trg_path", type=str, required=True, help="Đường dẫn lưu frames đầu ra")
    parser.add_argument("--distortion", type=float, default=0.6, help="Hệ số bóp méo")
    
    args = parser.parse_args()

    if not os.path.exists(args.trg_path):
        os.makedirs(args.trg_path)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    src_images_path = [
        os.path.join(args.src_path, f) 
        for f in os.listdir(args.src_path) 
        if f.lower().endswith(valid_extensions)
    ]

    if len(src_images_path) == 0:
        print("[LỖI] Không tìm thấy hình ảnh nào trong thư mục!")
    else:
        convert_all_images_circular(src_images_path, args.trg_path, args.distortion)