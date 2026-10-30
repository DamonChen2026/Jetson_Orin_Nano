#!/usr/bin/env python3
import argparse
import ctypes
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

import threading, queue

# 复用 DAY 2 已验证的 CUDA、预处理和后处理代码
# from day2_detect_image import CudaRuntime, letterbox, decode_output
from day2_detect_image import CudaRuntime, letterbox, nms_class_aware, COCO_NAMES

def make_gst_writer(path, fps, size):
    w, h = size
    pipeline = (
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc speed-preset=ultrafast tune=zerolatency bitrate=8000 threads=4 ! "
        "h264parse ! qtmux ! "
        f"filesink location={path}"
    )
    return cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)


class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, size, max_queue=16):
        # self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        self.writer = make_gst_writer(path, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")
        self.q = queue.Queue(maxsize=max_queue)   # 满了会阻塞,形成背压
        self.write_ms = []                         # 写线程自己的耗时
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            t0 = time.perf_counter()
            try:
                self.writer.write(frame)
            except Exception as e:
                self.err = e
                # 继续消费队列,避免主线程 put 时死锁
                while self.q.get() is not None:
                    pass
                break
            self.write_ms.append((time.perf_counter() - t0) * 1000)

    def write(self, frame):
        if self.err:
            raise self.err
        self.q.put(frame)       # cap.read() 每次返回新数组,不会被覆盖

    def release(self):
        t0 = time.perf_counter()
        self.q.put(None)
        self.t.join()
        self.writer.release()   # 这里会做 mp4 收尾,也要计入总时间
        self.drain_ms = (time.perf_counter() - t0) * 1000   # 排空+收尾耗时


class AsyncReader:
    def __init__(self, cap, max_queue=8):
        self.cap, self.q = cap, queue.Queue(max_queue)
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            ok, frame = self.cap.read()
            self.q.put(frame if ok else None)
            if not ok:
                break
    def read(self):
        frame = self.q.get()
        return (frame is not None), frame


def draw_detections(frame, detections):
    """在当前视频帧上绘制检测框和标签。"""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["score"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(y1, th + 4)

        cv2.rectangle(
            frame,
            (x1, text_y - th - 4),
            (x1 + tw, text_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p))


def decode_output_opt(output, ratio, pad_x, pad_y, original_w, original_h,
                  conf_threshold=0.25, iou_threshold=0.45):
    # YOLOv8 export output: [1, 84, 8400] -> [8400, 84]
    # pred = np.asarray(output, dtype=np.float32)
    # if pred.ndim == 3:
    #     pred = pred[0]
    # if pred.ndim != 2:
    #     raise RuntimeError(f"Unexpected output rank: {pred.shape}")

    # if pred.shape[0] < pred.shape[1] and pred.shape[0] in (84, 85):
    #     pred = pred.T
    # elif pred.shape[1] not in (84, 85):
    #     raise RuntimeError(
    #         f"Unexpected YOLOv8 output shape: {pred.shape}; expected [8400,84]"
    #     )
    pred = np.asarray(output)[0]                 # (84, 8400),先不转 float32
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected output rank: {pred.shape}")    
    # 统一成 (84, 8400) 布局。.T 只是改 stride 的视图,不拷贝数据
    if pred.shape[0] not in (84, 85):
        if pred.shape[1] in (84, 85):
            pred = pred.T
        else:
            raise RuntimeError(f"Unexpected YOLOv8 output shape: {pred.shape}")


    # YOLOv8 detection output has xywh + class scores (no objectness score).
    # class_scores = pred[:, 4:]
    # class_ids = np.argmax(class_scores, axis=1)
    # scores = class_scores[np.arange(len(pred)), class_ids]
    # mask = scores >= conf_threshold

    # pred = pred[mask]
    # scores = scores[mask]
    # class_ids = class_ids[mask]

    # if len(pred) == 0:
    #     return [], 0
    # 1) 在原布局上对 80 个类别取每列最大值:内存连续,速度快
    best = pred[4:].max(axis=0)                 # (8400,)
    mask = best >= conf_threshold
    candidate_count = int(mask.sum())
    if candidate_count == 0:
        return [], 0

    # 2) 只对候选框转置并转 float32(几十个框,几乎没有开销)
    cand = pred[:, mask].T.astype(np.float32)   # (N, 84)
    class_ids = cand[:, 4:].argmax(axis=1)
    scores = cand[np.arange(len(cand)), 4 + class_ids]

    # 3) xywh -> xyxy
    xywh = cand[:, :4]
    # xywh = pred[:, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # Undo letterbox padding and scale back to original image coordinates.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h - 1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

    keep = nms_class_aware(boxes, scores, class_ids, iou_threshold)
    detections = [
        {
            "box": boxes[i].round().astype(int).tolist(),
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
            "label": COCO_NAMES[int(class_ids[i])]
                if int(class_ids[i]) < len(COCO_NAMES)
                else str(int(class_ids[i])),
        }
        for i in keep
    ]
    return detections, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(
        description="DAY 3: YOLOv8n TensorRT video detection benchmark"
    )
    parser.add_argument("--engine", default="engines/yolov8n_fp16.engine")
    parser.add_argument("--video", default="videos/day3_long.mp4")
    parser.add_argument("--out-dir", default="out/day3")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA Graph for TensorRT inference",
    )    
    args = parser.parse_args()

    engine_path = Path(args.engine)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / f"{video_path.stem}_det.mp4"
    timing_csv = out_dir / "frame_timing.csv"
    report_path = out_dir / "experiment.txt"

    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. 初始化 TensorRT Engine 和执行上下文
    # ---------------------------------------------------------
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            f"Expected one input and one output: {input_names}, {output_names}"
        )

    input_name = input_names[0]
    output_name = output_names[0]
    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))

    if -1 in input_shape or -1 in output_shape:
        raise RuntimeError("This script expects static input/output shapes")
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise RuntimeError(f"Unexpected input shape: {input_shape}")

    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
    _, _, input_h, input_w = input_shape

    # ---------------------------------------------------------
    # 2. 打开视频并获取输入信息
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot decode the first video frame")

    # ---------------------------------------------------------
    # 3. 创建输出视频
    # ---------------------------------------------------------
    writer = AsyncVideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), source_fps if source_fps > 0 else 10.0, (frame_w, frame_h))
    # writer = cv2.VideoWriter(
    #     str(output_video),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     source_fps if source_fps > 0 else 10.0,
    #     (frame_w, frame_h),
    # )
    # if not writer.isOpened():
    #     cap.release()
    #     raise RuntimeError(f"Cannot create output video: {output_video}")

    # ---------------------------------------------------------
    # 4. 分配 CUDA 缓冲区，绑定 TensorRT 输入/输出地址
    # ---------------------------------------------------------
    cuda = CudaRuntime()
    host_output = np.empty(output_shape, dtype=output_dtype)

    d_input = None
    d_output = None
    stream = None

    graph = None
    graph_exec = None

    rows = []
    processed_frames = 0
    failed_reads = 0
    candidate_total = 0
    detection_total = 0
    timing_keys = [
        "read_ms",
        "preprocess_ms",
        "h2d_ms",
        "inference_ms",
        "d2h_ms",
        "postprocess_ms",
        "draw_ms",
        "write_ms",
        # "draw_write_ms",
    ]

    try:
        d_input = cuda.malloc(int(np.prod(input_shape)) * input_dtype.itemsize)
        d_output = cuda.malloc(host_output.nbytes)
        stream = cuda.create_stream()

        context.set_tensor_address(input_name, int(d_input.value))
        context.set_tensor_address(output_name, int(d_output.value))

        def preprocess(frame):
            padded, ratio, pad_x, pad_y = letterbox(frame, size=input_w)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[None, ...].astype(input_dtype) / 255.0
            )
            # tensor = cv2.dnn.blobFromImage(padded, 1/255.0, swapRB=True)   # 一步完成 BGR→RGB、归一化、NCHW
            # if input_dtype != np.float32:
            #     tensor = tensor.astype(input_dtype)
            if tuple(tensor.shape) != input_shape:
                raise RuntimeError(
                    f"Preprocessed shape {tensor.shape} != {input_shape}"
                )
            return tensor, ratio, pad_x, pad_y

        # 预热：使用首帧重复执行模型，不计入正式统计
        warm_tensor, _, _, _ = preprocess(first_frame)
        for _ in range(args.warmup):
            cuda.memcpy(
                d_input,
                __import__("ctypes").c_void_p(warm_tensor.ctypes.data),
                warm_tensor.nbytes,
                1,
            )
            ok = context.execute_async_v3(stream_handle=int(stream.value))
            if not ok:
                raise RuntimeError("TensorRT warmup inference failed")
            cuda.check(
                cuda.lib.cudaStreamSynchronize(stream),
                "warmup cudaStreamSynchronize",
            )

        # 回到视频开头，确保预热阶段不影响正式帧数
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        reader = AsyncReader(cap)             # 回到开头后再启动预取线程
        # CUDA Graph：只捕获 TensorRT 推理，不捕获 H2D/D2H
        if args.cuda_graph:
            print("Capturing CUDA Graph for TensorRT inference...")

            cuda.begin_capture(stream)
            try:
                ok = context.execute_async_v3(
                    stream_handle=int(stream.value)
                )
                if not ok:
                    raise RuntimeError(
                        "TensorRT inference failed during CUDA Graph capture"
                    )

                graph = cuda.end_capture(stream)

            except Exception:
                # 如果捕获过程中发生异常，尽力结束捕获，
                # 避免遗留未结束的 stream capture 状态。
                if graph is None:
                    try:
                        graph = cuda.end_capture(stream)
                    except Exception:
                        pass
                raise

            # 将捕获的 Graph 实例化为可重复执行的对象
            graph_exec = cuda.instantiate_graph(graph)

            # 等待捕获期间提交的工作完成
            cuda.check(
                cuda.lib.cudaStreamSynchronize(stream),
                "CUDA Graph capture synchronize",
            )

            print("CUDA Graph captured and instantiated successfully.")

        print("Engine:", engine_path)
        print("Video:", video_path)
        print("CUDA Runtime:", cuda.path)
        print("Input shape:", input_shape)
        print("Output shape:", output_shape)
        print(f"Video: {frame_w}x{frame_h}, source FPS={source_fps:.3f}")
        print("Container-reported frames:", reported_frames)
        print("Warmup iterations:", args.warmup)
        print("CUDA Graph enabled:", args.cuda_graph)
        print("Output video:", output_video)
        print("\nProcessing video...")

        # -----------------------------------------------------
        # 5. 逐帧处理并分别统计各阶段耗时
        # -----------------------------------------------------
        whole_loop_start = time.perf_counter()

        while True:
            read_start = time.perf_counter()
            ok, frame = reader.read() # cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000.0

            if not ok:
                break

            frame_start = time.perf_counter()

            t0 = time.perf_counter()
            tensor, ratio, pad_x, pad_y = preprocess(frame)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            cuda.memcpy(
                d_input,
                __import__("ctypes").c_void_p(tensor.ctypes.data),
                tensor.nbytes,
                1,
            )
            h2d_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            if args.cuda_graph:
                # 重放已经捕获并实例化的推理 Graph
                cuda.launch_graph(graph_exec, stream)
            else:
                # 保留原有普通 TensorRT 推理路径
                ok = context.execute_async_v3(
                    stream_handle=int(stream.value)
                )
                if not ok:
                    raise RuntimeError(
                        f"TensorRT inference failed at frame {processed_frames}"
                    )

            # 两种模式统一同步，保证后续 D2H 读取时推理已完成
            cuda.check(
                cuda.lib.cudaStreamSynchronize(stream),
                "inference cudaStreamSynchronize",
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            cuda.memcpy(
                __import__("ctypes").c_void_p(host_output.ctypes.data),
                d_output,
                host_output.nbytes,
                2,
            )
            d2h_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            detections, candidate_count = decode_output_opt(
                host_output,
                ratio,
                pad_x,
                pad_y,
                frame.shape[1],
                frame.shape[0],
                args.conf,
                args.iou,
            )
            postprocess_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            draw_detections(frame, detections)
            draw_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter();
            writer.write(frame)
            write_ms = (time.perf_counter() - t0) * 1000.0

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            processed_frames += 1
            candidate_total += candidate_count
            detection_total += len(detections)

            rows.append({
                "frame": processed_frames,
                "read_ms": read_ms,
                "preprocess_ms": preprocess_ms,
                "h2d_ms": h2d_ms,
                "inference_ms": inference_ms,
                "d2h_ms": d2h_ms,
                "postprocess_ms": postprocess_ms,
                "draw_ms": draw_ms,
                "write_ms": write_ms,
                "frame_processing_ms": frame_ms,
                "detections": len(detections),
                "candidates": candidate_count,
            })

            if processed_frames % 5 == 0:
                print(
                    f"Frames: {processed_frames}/{reported_frames or '?'} | "
                    f"last inference={inference_ms:.2f} ms | "
                    f"last frame={frame_ms:.2f} ms"
                )

        processing_wall_ms = (time.perf_counter() - whole_loop_start) * 1000.0

    finally:
        cap.release()
        writer.release()

        # 先销毁 Graph 执行实例，再销毁 Graph 本体
        if graph_exec is not None:
            cuda.destroy_graph_exec(graph_exec)

        if graph is not None:
            cuda.destroy_graph(graph)

        # 最后同步并释放 Stream 与设备内存
        if stream:
            cuda.check(
                cuda.lib.cudaStreamSynchronize(stream),
                "final cudaStreamSynchronize",
            )
            cuda.check(
                cuda.lib.cudaStreamDestroy(stream),
                "cudaStreamDestroy",
            )

        if d_output:
            cuda.free(d_output)

        if d_input:
            cuda.free(d_input)
    total_elapsed_s = time.perf_counter() - total_start

    # ---------------------------------------------------------
    # 6. 输出 CSV 和汇总报告
    # ---------------------------------------------------------
    if rows:
        with timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    actual_fps = (
        processed_frames / (processing_wall_ms / 1000.0)
        if processing_wall_ms > 0 else 0.0
    )

    lines = [
        "===== DAY 3 Video Detection Report =====",
        f"Engine: {engine_path}",
        f"Input video: {video_path}",
        f"Output video: {output_video}",
        f"Timing CSV: {timing_csv}",
        f"Input resolution: {frame_w}x{frame_h}",
        f"Source FPS: {source_fps:.3f}",
        f"Container-reported frames: {reported_frames}",
        f"Successfully processed frames: {processed_frames}",
        f"Failed frame reads at end: {failed_reads}",
        f"Confidence threshold: {args.conf}",
        f"IoU threshold: {args.iou}",
        f"Warmup iterations: {args.warmup}",
        f"CUDA Graph enabled: {args.cuda_graph}",
        f"Video processing loop wall time: {processing_wall_ms:.3f} ms",
        f"Actual processing FPS: {actual_fps:.3f}",
        f"Total elapsed time including initialization: {total_elapsed_s:.3f} s",
        f"Average candidates per frame: {candidate_total / max(1, processed_frames):.3f}",
        f"Average detections per frame: {detection_total / max(1, processed_frames):.3f}",
        f"Writer thread mean/p95 per frame: {np.mean(writer.write_ms):.3f} / {percentile(writer.write_ms, 95):.3f} ms",
        f"Writer drain+finalize after loop: {writer.drain_ms:.1f} ms",
        f"End-to-end FPS (loop + drain): {processed_frames / ((processing_wall_ms + writer.drain_ms) / 1000):.3f}",        
        "",
        "===== Per-stage timing (ms/frame) =====",
    ]

    for key in timing_keys:
        vals = [r[key] for r in rows]
        lines.append(
            f"{key:20s} mean={np.mean(vals):.3f} "
            f"median={np.median(vals):.3f} p95={percentile(vals, 95):.3f}"
            if vals else f"{key:20s} no data"
        )

    lines.extend([
        "",
        "Note: actual processing FPS uses the wall time of the video-processing loop.",
        "Per-stage timings are measured separately and may include CPU scheduling overhead.",
        "This is a short synthetic video made from still images, not a real camera stream.",
    ])

    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print("Saved report:", report_path)
    print("Saved per-frame timings:", timing_csv)


if __name__ == "__main__":
    main()
