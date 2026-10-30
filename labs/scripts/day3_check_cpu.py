import time
import cv2

cap = cv2.VideoCapture("videos/day3_test.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
w = cv2.VideoWriter("/tmp/t.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
ts = []
for f in frames * 4:          # 约 100 帧
    t0 = time.perf_counter(); w.write(f); ts.append((time.perf_counter() - t0) * 1000)
w.release()
print([round(x) for x in ts])