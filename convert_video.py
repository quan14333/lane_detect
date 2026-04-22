import cv2
import argparse
import os
from tqdm import tqdm
from utils import convert_image

def convert_video(src_video_path, out_video_path, distortion_coefficient=0.5, crop=True):
    # Kiểm tra file tồn tại
    if not os.path.exists(src_video_path):
        print(f"[LỖI] Không tìm thấy video: {src_video_path}")
        return

    # Khởi tạo đọc video
    cap = cv2.VideoCapture(src_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đọc frame đầu tiên trước để lấy kích thước đầu ra chuẩn xác
    # (Vì crop=True sẽ làm thay đổi kích thước frame gốc)
    ret, frame = cap.read()
    if not ret:
        print("[LỖI] Video trống hoặc không thể đọc.")
        return

    print("[INFO] Đang tính toán kích thước frame đầu ra...")
    first_distorted_frame = convert_image(frame, distortion_coefficient, crop)
    new_h, new_w, _ = first_distorted_frame.shape

    # Khởi tạo ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Định dạng mp4
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (new_w, new_h))

    # Ghi frame đầu tiên đã xử lý
    out.write(first_distorted_frame)

    # Xử lý các frame còn lại
    print(f"[INFO] Bắt đầu convert video. Tổng số frames: {total_frames}")
    for _ in tqdm(range(1, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Biến đổi frame
        distorted_frame = convert_image(frame, distortion_coefficient, crop)
        out.write(distorted_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    print(f"[INFO] Hoàn thành! Đã lưu video tại: {out_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert normal video to Fisheye video")
    parser.add_argument("--src_video", type=str, required=True, help="Đường dẫn đến video gốc")
    parser.add_argument("--out_video", type=str, default="output_fisheye.mp4", help="Đường dẫn video đầu ra")
    parser.add_argument("--distortion", type=float, default=0.5, help="Hệ số distortion (mặc định 0.5)")
    parser.add_argument("--crop", type=bool, default=True, help="Crop vùng đen (mặc định True)")

    args = parser.parse_args()
    
    convert_video(args.src_video, args.out_video, args.distortion, args.crop)