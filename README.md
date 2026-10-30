# Jetson_Orin_Nano
Notes on Learning with the Jetson Orin Nano Development Board

Board Base Info:
    chery@localhost:~$ jetson_release -v
    Software part of jetson-stats 7.2.2 - (c) 2026, Raffaello Bonghi
    Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super - Jetpack 7.2 GA [L4T 39.2.0]
    NV Power Mode[2]: MAXN_SUPER
    Serial Number: [XXX Show with: jetson_release -s XXX]
    Hardware:
    - 699-level Part Number: 699-13767-0005-300 K.2
    - P-Number: p3767-0005
    - Module: NVIDIA Jetson Orin Nano (Developer kit)
    - SoC: tegra234
    - CUDA Arch BIN: 8.7
    Platform:
    - Machine: aarch64
    - System: Linux
    - Distribution: Ubuntu 24.04 Noble Numbat
    - Release: 6.8.12-1021-tegra
    - Python: 3.12.3
    jtop:
    - Version: 7.2.2
    - Service: Active
    Libraries:
    - CUDA: 13.2.78
    - cuDNN: 9.20.0
    - TensorRT: 10.16.2.10
    - VPI: 4.1.3.0
    - Vulkan: 1.4.321
    - OpenCV: 4.8.0 - with CUDA: NO
    chery@localhost:~$ 
    chery@localhost:~$ cat /etc/nv_tegra_release 
    # R39 (release), REVISION: 2.0, GCID: 45755727, BOARD: generic, EABI: aarch64, DATE: Mon Jun  1 09:28:48 PM UTC 2026
    # KERNEL_VARIANT: oot
    TARGET_USERSPACE_LIB_DIR=nvidia
    TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia

isaac-ros-cli Folder contains the docker for Jetson
  user can run: bash isaac-ros-cli/start_dev.sh to create a docker for current user
  and then run: bash isaac-ros-cli/into_dev.sh to enter the docker

labs Folder for every test and result
