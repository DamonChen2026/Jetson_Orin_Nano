# Based on https://github.com/stereolabs/zed-docker

# Download dependencies for zed SDK installation RUN file
sudo apt-get update -y || true
sudo apt-get install --no-install-recommends lsb-release wget less zstd udev sudo apt-transport-https -y

# Download zed SDK installation RUN file to /tmp directory
cd /tmp

wget -q -O ZED_SDK_Linux.run https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.2/ZED_SDK_Tegra_L4T38.4_v5.2.0.zstd.run
chmod +x ./ZED_SDK_Linux.run
echo "Installing ZED SDK"
sudo ./ZED_SDK_Linux.run silent skip_od_module skip_python skip_drivers

# Cleanup
rm ZED_SDK_Linux.run
sudo rm -rf /var/lib/apt/lists/*
