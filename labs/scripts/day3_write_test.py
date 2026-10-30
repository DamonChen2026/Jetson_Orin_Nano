import cv2, time, numpy as np
frame = cv2.imread("images/bus.jpg")
frame = cv2.resize(frame, (1280, 720))
for name, fourcc, ext in [("mp4v", "mp4v", "mp4"), ("MJPG", "MJPG", "avi")]:
    w = cv2.VideoWriter(f"/tmp/t.{ext}", cv2.VideoWriter_fourcc(*fourcc), 10, (1280, 720))
    ts = []
    for _ in range(100):
        t0 = time.perf_counter(); w.write(frame); ts.append((time.perf_counter() - t0) * 1000)
    w.release()
    print(name, "mean", np.mean(ts), "p95", np.percentile(ts, 95))