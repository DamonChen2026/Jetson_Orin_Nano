import glob
import os
import cv2

IMAGE_DIR = "images"
OUTPUT_PATH = "videos/day3_test.mp4"
TARGET_FPS = 10.0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

if not paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    TARGET_FPS,
    (TARGET_WIDTH, TARGET_HEIGHT),
)

if not writer.isOpened():
    raise RuntimeError(
        "VideoWriter failed to open. "
        "Check OpenCV/FFmpeg video encoding support."
    )

written = 0

try:
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Skip unreadable image: {path}")
            continue

        h, w = image.shape[:2]
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)

        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        frame = __import__("numpy").zeros(
            (TARGET_HEIGHT, TARGET_WIDTH, 3),
            dtype="uint8",
        )

        x = (TARGET_WIDTH - new_w) // 2
        y = (TARGET_HEIGHT - new_h) // 2
        frame[y:y + new_h, x:x + new_w] = resized

        writer.write(frame)
        written += 1
        print(f"[{written:02d}/{len(paths):02d}] {path}")

finally:
    writer.release()

if written == 0:
    raise RuntimeError("No frames were written")

cap = cv2.VideoCapture(OUTPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot reopen output video: {OUTPUT_PATH}")

reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
reported_fps = cap.get(cv2.CAP_PROP_FPS)
reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

decoded = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    decoded += 1

cap.release()

print("\n===== Video verification =====")
print("Output:", OUTPUT_PATH)
print("Frames written:", written)
print("Frames reported by container:", reported_frames)
print("Frames decoded successfully:", decoded)
print("FPS:", reported_fps)
print("Resolution:", f"{reported_width}x{reported_height}")
print("Expected duration (s):", written / TARGET_FPS)
print("File size (bytes):", os.path.getsize(OUTPUT_PATH))

if decoded != written:
    raise RuntimeError(f"Frame count mismatch: written={written}, decoded={decoded}")