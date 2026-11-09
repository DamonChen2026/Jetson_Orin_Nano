
#!/usr/bin/env python3
"""
启动本脚本前，请确保已启动 ROS2 节点发布图像，例如：
    ros2 launch realsense2_camera rs_launch.py
"""
import argparse
import threading
import time
from pathlib import Path
import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import tensorrt as trt
import numpy as np
from day2_detect_image import CudaRuntime, letterbox
from day3_video_detect import draw_detections, decode_output_opt, AsyncVideoWriter


class TensorRTResources:
    """TensorRT Engine、Execution Context 和 CUDA 资源管理。"""

    def __init__(self, engine_path):
        self.engine_path = engine_path

        self.runtime = None
        self.engine = None
        self.context = None

        self.cuda = None
        self.d_input = None
        self.d_output = None
        self.stream = None

        self.host_output = None

        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    def _initialize(self):
        print("\n===== TensorRT Initialization =====")

        # 1. 创建 Runtime 并加载 Engine
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        engine_bytes = self.engine_path.read_bytes()

        self.engine = self.runtime.deserialize_cuda_engine(
            engine_bytes
        )

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT Engine: {self.engine_path}"
            )

        print("Engine loaded:", self.engine_path)

        # 2. 创建 Execution Context
        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT Execution Context"
            )

        # 3. 查询输入输出 Tensor
        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)

            print(
                f"Tensor: {name}, "
                f"mode={mode}, shape={shape}, dtype={dtype}"
            )

            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected one input and one output, "
                f"got input={input_names}, output={output_names}"
            )

        self.input_name = input_names[0]
        self.output_name = output_names[0]

        self.input_shape = tuple(
            self.engine.get_tensor_shape(self.input_name)
        )

        self.output_shape = tuple(
            self.engine.get_tensor_shape(self.output_name)
        )

        if -1 in self.input_shape or -1 in self.output_shape:
            raise RuntimeError(
                "Dynamic input/output shape is not supported"
            )

        if (
            len(self.input_shape) != 4
            or self.input_shape[0] != 1
            or self.input_shape[1] != 3
        ):
            raise RuntimeError(
                f"Unexpected input shape: {self.input_shape}"
            )

        self.input_dtype = np.dtype(
            trt.nptype(
                self.engine.get_tensor_dtype(self.input_name)
            )
        )

        self.output_dtype = np.dtype(
            trt.nptype(
                self.engine.get_tensor_dtype(self.output_name)
            )
        )

        _, _, self.input_h, self.input_w = self.input_shape

        # 4. 创建 CPU 输出数组
        self.host_output = np.empty(
            self.output_shape,
            dtype=self.output_dtype,
        )

        # 5. 创建 CUDA Runtime
        self.cuda = CudaRuntime()

        print("CUDA Runtime:", self.cuda.path)

        input_bytes = (
            int(np.prod(self.input_shape))
            * self.input_dtype.itemsize
        )

        output_bytes = self.host_output.nbytes

        # 6. 分配 GPU 显存
        self.d_input = self.cuda.malloc(input_bytes)
        self.d_output = self.cuda.malloc(output_bytes)

        print("GPU input buffer:", input_bytes, "bytes")
        print("GPU output buffer:", output_bytes, "bytes")

        # 7. 创建 CUDA Stream
        self.stream = self.cuda.create_stream()

        # 8. 将 GPU 显存地址绑定到 TensorRT Tensor
        if not self.context.set_tensor_address(
            self.input_name,
            int(self.d_input.value),
        ):
            raise RuntimeError("Failed to bind TensorRT input address")

        if not self.context.set_tensor_address(
            self.output_name,
            int(self.d_output.value),
        ):
            raise RuntimeError("Failed to bind TensorRT output address")

        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_shape)
        print("Input dtype:", self.input_dtype)
        print("Output dtype:", self.output_dtype)
        print("CUDA Stream created.")
        print("TensorRT initialization completed.")
        print("===================================\n")

    def close(self):
        """释放 CUDA 资源。"""

        if self.cuda is not None:
            if self.stream is not None:
                self.cuda.check(
                    self.cuda.lib.cudaStreamSynchronize(self.stream),
                    "final cudaStreamSynchronize",
                )
                self.cuda.check(
                    self.cuda.lib.cudaStreamDestroy(self.stream),
                    "cudaStreamDestroy",
                )
                self.stream = None

            if self.d_output is not None:
                self.cuda.free(self.d_output)
                self.d_output = None

            if self.d_input is not None:
                self.cuda.free(self.d_input)
                self.d_input = None

        self.context = None
        self.engine = None
        self.runtime = None

        print("TensorRT resources released.")


class LatestFrameSubscriber(Node):
    """ROS2 接收线程：接收图像并更新 Latest Frame 单槽缓存。"""

    def __init__(self, topic):
        super().__init__("day4_latest_frame_subscriber")

        self.bridge = CvBridge()

        # 共享数据：接收线程写入，主线程读取
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.latest_frame = None

        # 累计统计数据
        self.received_frames = 0
        self.overwritten_frames = 0
        self.callback_ms_sum = 0.0
        self.callback_ms_max = 0.0
        self.raw_img_width = 0
        self.raw_img_height = 0
        self.last_report_time = 0
        self.fps = 0.0

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Subscribed to: {topic}")

    def image_callback(self, msg):
        """ROS2 callback：只接收、转换和更新缓存，不执行推理。"""

        t0 = time.perf_counter()
        if self.last_report_time != 0:
            elapsed = t0 - self.last_report_time
            self.fps = 1.0 / elapsed
        self.last_report_time = t0

        try:
            # ROS Image -> OpenCV BGR8
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            # 独立持有图像数据，避免共享缓冲区生命周期问题
            frame = frame.copy()

            callback_ms = (time.perf_counter() - t0) * 1000.0
            self.raw_img_width = msg.width
            self.raw_img_height = msg.height
            # 锁只保护共享数据更新，不在锁内执行耗时操作
            with self.lock:
                if self.latest_frame is not None:
                    self.overwritten_frames += 1

                self.latest_frame = frame
                self.received_frames += 1

                self.callback_ms_sum += callback_ms
                self.callback_ms_max = max(
                    self.callback_ms_max,
                    callback_ms,
                )

                # 通知主线程有新帧
                self.frame_event.set()

        except Exception as exc:
            self.get_logger().error(
                f"Image conversion failed: {exc}"
            )

    def take_latest_frame(self):
        """主线程取出最新帧，取出后清空单槽缓存。"""

        with self.lock:
            frame = self.latest_frame
            self.latest_frame = None

            # 与缓存操作使用同一把锁，避免丢失唤醒信号
            self.frame_event.clear()

        return frame

    def get_stats(self):
        """读取统计数据快照。"""

        with self.lock:
            return {
                "received": self.received_frames,
                "overwritten": self.overwritten_frames,
                "callback_ms_sum": self.callback_ms_sum,
                "callback_ms_max": self.callback_ms_max,
                "raw_img_width": self.raw_img_width,
                "raw_img_height": self.raw_img_height,
                "fps": self.fps,
            }

# def preprocess(frame):
#     padded, ratio, pad_x, pad_y = letterbox(frame, size=node.input_shape)
#     rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
#     tensor = np.ascontiguousarray(
#         rgb.transpose(2, 0, 1)[None, ...].astype(input_dtype) / 255.0
#     )

#     if tuple(tensor.shape) != input_shape:
#         raise RuntimeError(
#             f"Preprocessed shape {tensor.shape} != {input_shape}"
#         )
#     return tensor, ratio, pad_x, pad_y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        default="engines/yolov8n_fp16.engine",
    )
    parser.add_argument("--out-dir", default="out/day4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--write", type=bool, default=False)
    args = parser.parse_args()

    engine_path = Path(args.engine)

    if not engine_path.is_file():
        raise FileNotFoundError(
            f"Engine not found: {engine_path}"
        )

    # TensorRT 资源由主线程初始化和持有
    trt_resources = TensorRTResources(engine_path)

    # ---------------------------------------------------------
    # 创建输出视频
    # ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / f"RealSense_det.mp4"    

    topic = "/camera/camera/color/image_raw"

    rclpy.init()

    node = LatestFrameSubscriber(topic)

    # ROS2 Executor 在独立线程运行
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(
        target=executor.spin,
        name="ros2-executor",
        daemon=True,
    )
    ros_thread.start()

    # 主线程负责取帧和显示
    window_name = "DAY4-3 Latest Frame"
    window_name_det = "DAY4-3 Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_det, cv2.WINDOW_NORMAL)

    displayed_frames = 0

    window_start = time.perf_counter()
    last_received = 0
    last_overwritten = 0
    last_callback_sum = 0.0
    writer = None
    print("DAY4-3 Step 1: ROS2 Latest Frame")
    print("Press Q or ESC to exit.")

    try:
        while rclpy.ok():

            frame = node.take_latest_frame()
            def preprocess(frame):
                padded, ratio, pad_x, pad_y = letterbox(frame, size=trt_resources.input_w)
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                tensor = np.ascontiguousarray(
                    rgb.transpose(2, 0, 1)[None, ...].astype(trt_resources.input_dtype) / 255.0
                )

                if tuple(tensor.shape) != trt_resources.input_shape:
                    raise RuntimeError(
                        f"Preprocessed shape {tensor.shape} != {trt_resources.input_shape}"
                    )
                return tensor, ratio, pad_x, pad_y

            if frame is not None:
                if displayed_frames == 0:
                    print(
                        f"Raw image size: {node.raw_img_width}x{node.raw_img_height}"
                    )
                    writer = AsyncVideoWriter(output_video,
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            node.fps if node.fps > 0 else 30.0, 
                            (node.raw_img_width, node.raw_img_height))

                displayed_frames += 1

                cv2.putText(
                    frame,
                    f"Displayed: {displayed_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(window_name, frame)

                tensor, ratio, pad_x, pad_y = preprocess(frame)
                # 拷贝
                trt_resources.cuda.memcpy(
                    trt_resources.d_input,
                    __import__("ctypes").c_void_p(tensor.ctypes.data),
                    tensor.nbytes,
                    1,
                )

                # 保留原有普通 TensorRT 推理路径
                ok = trt_resources.context.execute_async_v3(
                    stream_handle=int(trt_resources.stream.value)
                )
                if not ok:
                    raise RuntimeError(
                        f"TensorRT inference failed at frame {processed_frames}"
                    )

                # 两种模式统一同步，保证后续 D2H 读取时推理已完成
                trt_resources.cuda.check(
                    trt_resources.cuda.lib.cudaStreamSynchronize(trt_resources.stream),
                    "inference cudaStreamSynchronize",
                )

                trt_resources.cuda.memcpy(
                    __import__("ctypes").c_void_p(trt_resources.host_output.ctypes.data),
                    trt_resources.d_output,
                    trt_resources.host_output.nbytes,
                    2,
                )

                detections, candidate_count = decode_output_opt(
                    trt_resources.host_output,
                    ratio,
                    pad_x,
                    pad_y,
                    frame.shape[1],
                    frame.shape[0],
                    args.conf,
                    args.iou,
                )

                draw_detections(frame, detections)

                cv2.putText(
                    frame,
                    f"Displayed: {displayed_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name_det, frame)
                if writer is not None:
                    if args.write:
                        writer.write(frame)
                else:
                    print("Writer is not initialized.")

            else:
                # 没有新帧时等待，避免空转占满 CPU
                node.frame_event.wait(timeout=0.01)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            # 每秒统计一次
            now = time.perf_counter()
            elapsed = now - window_start

            if elapsed >= 1.0:
                stats = node.get_stats()

                received_delta = (
                    stats["received"] - last_received
                )
                overwritten_delta = (
                    stats["overwritten"] - last_overwritten
                )
                callback_sum_delta = (
                    stats["callback_ms_sum"] - last_callback_sum
                )

                receive_fps = received_delta / elapsed

                callback_avg = (
                    callback_sum_delta / received_delta
                    if received_delta > 0 else 0.0
                )

                print(
                    f"Receive FPS={receive_fps:.2f}, "
                    f"Received={stats['received']}, "
                    f"Displayed={displayed_frames}, "
                    f"Overwritten={stats['overwritten']} "
                    f"(+{overwritten_delta}), "
                    f"Callback avg={callback_avg:.3f} ms, "
                    f"Callback max={stats['callback_ms_max']:.3f} ms"
                )

                last_received = stats["received"]
                last_overwritten = stats["overwritten"]
                last_callback_sum = stats["callback_ms_sum"]
                window_start = now

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping DAY4-3...")

        executor.shutdown()
        ros_thread.join(timeout=2.0)

        executor.remove_node(node)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()

        trt_resources.close()
        if writer is not None:
            writer.release()
        print("DAY4-3 stopped.")


if __name__ == "__main__":
    main()
