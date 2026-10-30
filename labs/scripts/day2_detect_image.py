#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class CudaRuntime:
    """Minimal CUDA Runtime wrapper; no PyCUDA/cuda-python dependency."""

    def __init__(self):
        candidates = [
            ctypes.util.find_library("cudart"),
            "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/aarch64-linux-gnu/libcudart.so",
            "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
        ]
        self.lib = None
        for path in candidates:
            if not path:
                continue
            try:
                self.lib = ctypes.CDLL(path)
                self.path = path
                break
            except OSError:
                pass

        if self.lib is None:
            raise RuntimeError(
                "Cannot load libcudart.so. Check CUDA Runtime library paths."
            )

        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
    
        """
        以下修订是增加 CUDA Graph 功能
        """
        # CUDA Graph API
        self.lib.cudaStreamBeginCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.cudaStreamBeginCapture.restype = ctypes.c_int

        self.lib.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamEndCapture.restype = ctypes.c_int

        self.lib.cudaGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        self.lib.cudaGraphInstantiateWithFlags.restype = ctypes.c_int

        self.lib.cudaGraphLaunch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cudaGraphLaunch.restype = ctypes.c_int

        self.lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphDestroy.restype = ctypes.c_int

        self.lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphExecDestroy.restype = ctypes.c_int

    def check(self, code, operation):
        if code != 0:
            raise RuntimeError(f"{operation} failed: CUDA error code {code}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def free(self, ptr):
        if ptr:
            self.check(self.lib.cudaFree(ptr), "cudaFree")

    def memcpy(self, dst, src, nbytes, kind):
        self.check(
            self.lib.cudaMemcpy(dst, src, nbytes, kind),
            f"cudaMemcpy(kind={kind})",
        )

    def create_stream(self):
        stream = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreate(ctypes.byref(stream)),
            "cudaStreamCreate",
        )
        return stream

    """
    以下修订是增加 CUDA Graph 功能
    """

    def begin_capture(self, stream):
        # 0 表示 cudaStreamCaptureModeGlobal
        self.check(
            self.lib.cudaStreamBeginCapture(stream, 0),
            "cudaStreamBeginCapture",
        )

    def end_capture(self, stream):
        graph = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamEndCapture(
                stream, ctypes.byref(graph)
            ),
            "cudaStreamEndCapture",
        )
        if not graph.value:
            raise RuntimeError("CUDA Graph capture returned a null graph")
        return graph

    def instantiate_graph(self, graph):
        graph_exec = ctypes.c_void_p()
        self.check(
            self.lib.cudaGraphInstantiateWithFlags(
                ctypes.byref(graph_exec), graph, 0
            ),
            "cudaGraphInstantiateWithFlags",
        )
        if not graph_exec.value:
            raise RuntimeError("CUDA Graph instantiation returned a null handle")
        return graph_exec

    def launch_graph(self, graph_exec, stream):
        self.check(
            self.lib.cudaGraphLaunch(graph_exec, stream),
            "cudaGraphLaunch",
        )

    def destroy_graph(self, graph):
        if graph and graph.value:
            self.check(
                self.lib.cudaGraphDestroy(graph),
                "cudaGraphDestroy",
            )

    def destroy_graph_exec(self, graph_exec):
        if graph_exec and graph_exec.value:
            self.check(
                self.lib.cudaGraphExecDestroy(graph_exec),
                "cudaGraphExecDestroy",
            )


def letterbox(image, size=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    result = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return result, ratio, left, top


def nms_class_aware(boxes, scores, class_ids, iou_threshold=0.45):
    keep = []
    for cls in np.unique(class_ids):
        indices = np.where(class_ids == cls)[0]
        if len(indices) == 0:
            continue

        xyxy = boxes[indices]
        cls_scores = scores[indices]
        xywh = np.column_stack((
            xyxy[:, 0],
            xyxy[:, 1],
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1],
        ))
        selected = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            cls_scores.astype(float).tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        if len(selected):
            keep.extend(indices[np.asarray(selected).reshape(-1)].tolist())

    return sorted(keep, key=lambda i: float(scores[i]), reverse=True)


def decode_output(output, ratio, pad_x, pad_y, original_w, original_h,
                  conf_threshold=0.25, iou_threshold=0.45):
    # YOLOv8 export output: [1, 84, 8400] -> [8400, 84]
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected output rank: {pred.shape}")

    if pred.shape[0] < pred.shape[1] and pred.shape[0] in (84, 85):
        pred = pred.T
    elif pred.shape[1] not in (84, 85):
        raise RuntimeError(
            f"Unexpected YOLOv8 output shape: {pred.shape}; expected [8400,84]"
        )

    # YOLOv8 detection output has xywh + class scores (no objectness score).
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(pred)), class_ids]
    mask = scores >= conf_threshold

    pred = pred[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(pred) == 0:
        return [], 0

    xywh = pred[:, :4]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="engines/yolov8n_fp16.engine")
    parser.add_argument("--image", default="images/bus.jpg")
    parser.add_argument("--out-dir", default="out/detect")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    image_path = Path(args.image)
    engine_path = Path(args.engine)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 加载图片
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    #
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    input_names = []
    output_names = []
    print("Engine:", engine_path)
    print("Image:", image_path)
    print("Original image:", image.shape[1], "x", image.shape[0])

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        print(f"Tensor {name}: mode={mode}, shape={shape}, dtype={dtype}")
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            f"Expected one input and one output; got {input_names}, {output_names}"
        )

    input_name, output_name = input_names[0], output_names[0]
    input_shape = tuple(engine.get_tensor_shape(input_name))
    if -1 in input_shape:
        raise RuntimeError(f"Dynamic input shape not handled in this script: {input_shape}")
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise RuntimeError(f"Expected NCHW batch-1 input, got {input_shape}")

    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
    _, channels, input_h, input_w = input_shape
    if channels != 3:
        raise RuntimeError(f"Expected 3-channel input, got {input_shape}")

    # add time info 
    pipeline_start = time.perf_counter()
    preprocess_start = time.perf_counter()

    padded, ratio, pad_x, pad_y = letterbox(image, size=input_w)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # 参数中transpose， 改变列顺序，原来的 (hwc) -> (CHW), 
    # 接着 增加一列 -> (NCHW)
    host_input = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None, ...].astype(input_dtype) / 255.0 )

    # add time info 
    preprocess_ms = ( time.perf_counter() - preprocess_start ) * 1000.0

    if tuple(host_input.shape) != input_shape:
        raise RuntimeError(
            f"Preprocessed shape {host_input.shape} != engine input {input_shape}"
        )

    output_shape = tuple(engine.get_tensor_shape(output_name))
    if any(d < 0 for d in output_shape):
        raise RuntimeError(f"Dynamic output shape not handled: {output_shape}")
    host_output = np.empty(output_shape, dtype=output_dtype)

    cuda = CudaRuntime()
    print("CUDA Runtime:", cuda.path)
    d_input = d_output = None
    stream = None
    try:
        d_input = cuda.malloc(host_input.nbytes)
        d_output = cuda.malloc(host_output.nbytes)
        stream = cuda.create_stream()

        context.set_tensor_address(input_name, int(d_input.value))
        context.set_tensor_address(output_name, int(d_output.value))

        # add time info  CPU -> GPU
        t0 = time.perf_counter()

        # cudaMemcpy kind: 1 = HostToDevice, 2 = DeviceToHost
        cuda.memcpy(
            d_input, ctypes.c_void_p(host_input.ctypes.data),
            host_input.nbytes, 1
        )
        h2d_ms = (time.perf_counter() - t0) * 1000.0

        # add time info  TensorRT 推理：包含提交任务及等待 GPU 完成的时间
        # t0 = time.perf_counter()
        # ok = context.execute_async_v3(stream_handle=int(stream.value))
        # if not ok:
        #     raise RuntimeError("TensorRT execute_async_v3 returned False")

        # cuda.check(
        #     cuda.lib.cudaStreamSynchronize(stream),
        #     "cudaStreamSynchronize",
        # )

        # inference_ms = (time.perf_counter() - t0) * 1000.0

        # 预热 10 次，不计入统计
        for _ in range(10):
            context.execute_async_v3(stream_handle=int(stream.value))
            cuda.lib.cudaStreamSynchronize(stream)

        # 正式测试 50 次
        inference_times = []

        for _ in range(50):
            t0 = time.perf_counter()

            context.execute_async_v3(stream_handle=int(stream.value))
            cuda.lib.cudaStreamSynchronize(stream)
            # time.sleep(0.04)      # 确认GPU在两次推理之间空闲，频率/电源状态降档（DVFS）
                                    # 前面的GPU预热失效，导致GPU推理耗时增加
            inference_times.append( (time.perf_counter() - t0) * 1000 )

        times = sorted(inference_times)

        print(f"TensorRT mean:   {sum(times) / len(times):.3f} ms")
        print(f"TensorRT min:    {times[0]:.3f} ms")
        print(f"TensorRT median: {times[len(times) // 2]:.3f} ms")
        print(f"TensorRT p95:    {times[int(len(times) * 0.95) - 1]:.3f} ms")        

        # add time info  GPU -> CPU
        t0 = time.perf_counter()

        cuda.memcpy(
            ctypes.c_void_p(host_output.ctypes.data), d_output,
            host_output.nbytes, 2
        )
        d2h_ms = (time.perf_counter() - t0) * 1000.0

        # add time info  后处理
        postprocess_start = time.perf_counter()
        detections, candidate_count = decode_output(
            host_output, ratio, pad_x, pad_y,
            image.shape[1], image.shape[0],
            args.conf, args.iou,
        )
        postprocess_ms = ( time.perf_counter() - postprocess_start ) * 1000.0

        # add time info  可视化
        visualize_start = time.perf_counter()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f'{det["label"]} {det["score"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            text_y = max(y1, th + 4)
            cv2.rectangle(
                image, (x1, text_y - th - 4),
                (x1 + tw, text_y + baseline), (0, 255, 0), -1
            )
            cv2.putText(
                image, label, (x1, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                cv2.LINE_AA
            )

        out_path = out_dir / f"{image_path.stem}_det.jpg"
        if not cv2.imwrite(str(out_path), image):
            raise RuntimeError(f"Failed to save output: {out_path}")

        print(f"Pre-NMS candidates (conf >= {args.conf}): {candidate_count}")
        print(f"Final detections: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(
                f'[{i:02d}] {det["label"]:16s} '
                f'conf={det["score"]:.3f} box={det["box"]}'
            )

        visualize_save_ms = ( time.perf_counter() - visualize_start ) * 1000.0

        pipeline_ms = ( time.perf_counter() - pipeline_start ) * 1000.0

        print("\n===== Timing (ms) =====")
        print(f"Preprocess:          {preprocess_ms:.3f}")
        print(f"Host to Device:      {h2d_ms:.3f}")
        # print(f"TensorRT inference:  {inference_ms:.3f}")
        print(f"Device to Host:      {d2h_ms:.3f}")
        print(f"Postprocess + NMS:   {postprocess_ms:.3f}")
        print(f"Draw + save image:   {visualize_save_ms:.3f}")
        print(f"Pipeline total:      {pipeline_ms:.3f}")            
        print("Saved:", out_path)


    finally:
        if stream:
            cuda.check(cuda.lib.cudaStreamSynchronize(stream), "final sync")
            cuda.check(cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")
        if d_output:
            cuda.free(d_output)
        if d_input:
            cuda.free(d_input)


if __name__ == "__main__":
    main()

