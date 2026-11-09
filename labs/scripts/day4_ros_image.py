
"""ROS 2 RealSense 彩色图像查看器。

ROS 2 环境接入（Ubuntu 示例）：
1. 安装依赖：
    sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-sensor-msgs \
         python3-opencv
2. 在运行脚本的终端加载 ROS 2 环境：
    source /opt/ros/$ROS_DISTRO/setup.bash
    若驱动来自 ROS 2 工作空间，还需 source <工作空间>/install/setup.bash。
3. 启动 RealSense 相机驱动，确认主题存在且类型为
    sensor_msgs/msg/Image：
    ros2 topic list
    ros2 topic info /camera/camera/color/image_raw
4. 运行：python3 day4_ros_image.py

本脚本直接创建 ROS 2 节点，因此无需先将它打包成 ROS 2 包。若要在
无图形桌面的设备上运行，需要去掉或替换 OpenCV 的 imshow/waitKey 显示。
"""

import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RealSenseImageViewer(Node):
    """订阅 RealSense 彩色图像、显示画面并报告接收性能。

    ``__init__`` 将该对象注册为名为 ``realsense_image_viewer`` 的 ROS 2
    节点，设置图像主题和 CvBridge，并以 ``qos_profile_sensor_data`` 创建
    ``sensor_msgs/msg/Image`` 订阅。传感器 QoS 适合实时数据流；相机驱动的
    主题名或 QoS 不同时，需要调整 ``self.topic`` 或订阅 QoS。

    每收到一帧，``image_callback`` 先用 CvBridge 转成 OpenCV 的 BGR 图像，
    再记录回调到达间隔、显示窗口并计算回调耗时。每 5 秒输出 FPS、帧数、
    分辨率、消息编码、平均回调耗时和平均到达间隔。耗时由本机
    ``time.perf_counter`` 测量，不代表相机时间戳或端到端延迟。报告后会
    清空当前窗口的统计列表和帧计数。

    ``main`` 初始化 ROS 2、通过 ``rclpy.spin`` 持续处理消息，并在退出时
    销毁节点、关闭 OpenCV 窗口及 ROS 2。按 Ctrl+C 可停止程序。
    """
    def __init__(self):
        super().__init__("realsense_image_viewer")

        self.topic = "/camera/camera/color/image_raw"
        self.bridge = CvBridge()

        self.frame_count = 0
        self.last_report_time = time.perf_counter()
        self.last_frame_time = None

        self.callback_times = []
        self.frame_intervals = []

        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Subscribing to: {self.topic}"
        )

    def image_callback(self, msg):
        callback_start = time.perf_counter()

        # 1. ROS Image -> OpenCV BGR
        cv_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        # 2. Measure callback arrival interval
        now = time.perf_counter()

        if self.last_frame_time is not None:
            interval_ms = (now - self.last_frame_time) * 1000
            self.frame_intervals.append(interval_ms)

        self.last_frame_time = now

        # 3. Display image
        cv2.imshow("RealSense D435 RGB", cv_image)
        cv2.waitKey(1)

        # 4. Count frames and callback cost
        callback_ms = (time.perf_counter() - callback_start) * 1000

        self.callback_times.append(callback_ms)
        self.frame_count += 1

        # 5. Report statistics every 5 seconds
        report_now = time.perf_counter()
        elapsed = report_now - self.last_report_time

        if elapsed >= 5.0:
            fps = self.frame_count / elapsed

            avg_callback = (
                sum(self.callback_times) / len(self.callback_times)
            )

            avg_interval = (
                sum(self.frame_intervals) / len(self.frame_intervals)
                if self.frame_intervals else 0.0
            )

            self.get_logger().info(
                f"FPS={fps:.2f}, "
                f"frames={self.frame_count}, "
                f"image={msg.width}x{msg.height}, "
                f"encoding={msg.encoding}, "
                f"callback_avg={avg_callback:.3f} ms, "
                f"arrival_interval_avg={avg_interval:.3f} ms"
            )

            self.frame_count = 0
            self.callback_times.clear()
            self.frame_intervals.clear()
            self.last_report_time = report_now


def main(args=None):
    rclpy.init(args=args)

    node = RealSenseImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
