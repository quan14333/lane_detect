import cv2
from ifisheye import convert_image
from tqdm import tqdm

def convert_video_to_fisheye(
    input_path,
    output_path,
    distortion_coefficient=0.5,
    crop=True
):
    cap = cv2.VideoCapture(input_path)

    # Lấy info video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đọc frame đầu để biết size sau khi fisheye
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    test_frame = convert_image(frame, distortion_coefficient, crop)
    new_h, new_w = test_frame.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    # Reset lại video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[INFO] Processing video...")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # 👉 APPLY fisheye giống hệt code của bạn
        new_frame = convert_image(frame, distortion_coefficient, crop)

        out.write(new_frame)

    cap.release()
    out.release()
    print("[INFO] Done!")

convert_video_to_fisheye(
    input_path="input.mp4",
    output_path="output_fisheye.mp4",
    distortion_coefficient=0.5,  # giống code bạn
    crop=True
)