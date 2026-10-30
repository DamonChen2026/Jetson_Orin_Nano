#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

echo "Creating non-root container '${USERNAME}' for host user uid=${HOST_USER_UID}:gid=${HOST_USER_GID}"

if [ ! "$(getent group "${HOST_USER_GID}")" ]; then
  groupadd --gid "${HOST_USER_GID}" "${USERNAME}" &>/dev/null
else
  CONFLICTING_GROUP_NAME=$(getent group "${HOST_USER_GID}" | cut -d: -f1)
  groupmod -o --gid "${HOST_USER_GID}" -n "${USERNAME}" "${CONFLICTING_GROUP_NAME}"
fi

if [ ! "$(getent passwd "${HOST_USER_UID}")" ]; then
  useradd --no-log-init --uid "${HOST_USER_UID}" --gid "${HOST_USER_GID}" -m "${USERNAME}" &>/dev/null
else
  CONFLICTING_USER_NAME=$(getent passwd "${HOST_USER_UID}" | cut -d: -f1)
  usermod -l "${USERNAME}" -u "${HOST_USER_UID}" -g "${HOST_USER_GID}" -m -d /home/"${USERNAME}" "${CONFLICTING_USER_NAME}" &>/dev/null
  mkdir -p /home/"${USERNAME}"
  # Add default user bash files (ensures shell works properly and autocompletes)
  cp /etc/skel/.[^.]* /home/"${USERNAME}"/
  chown "${USERNAME}":"${USERNAME}" /home/"${USERNAME}"/.bashrc /home/"${USERNAME}"/.profile \
    /home/"${USERNAME}"/.bash_logout
  # Wipe files that may create issues for users with large uid numbers.
  rm -f /var/log/lastlog /var/log/faillog
fi

# Update 'admin' user
chown "${USERNAME}":"${USERNAME}" /home/"${USERNAME}"
echo "${USERNAME}" ALL=\(root\) NOPASSWD:ALL >/etc/sudoers.d/"${USERNAME}"
chmod 0440 /etc/sudoers.d/"${USERNAME}"
adduser "${USERNAME}" video >/dev/null
adduser "${USERNAME}" plugdev >/dev/null
adduser "${USERNAME}" sudo >/dev/null

# The FSYNC node is root:root by default, so unlike the group-owned nodes below it
# has no group to add the user to. So here we grant access by changing the node
# itself to video group.
# The SIPL camera node uses it to control the GMSL cameras' frame-sync generator.
# Note: under --privileged /dev is the host's, so this persists on the host until
# reboot.
if [ -e /dev/fsync-group ]; then
  if ! chgrp video /dev/fsync-group || ! chmod 0660 /dev/fsync-group; then
    echo "Warning: could not grant the video group access to /dev/fsync-group" >&2
  fi
fi

# Ensure ZED SDK files are owned by the workspace user (if installed at build)
if [ -d /usr/local/zed ]; then
  chown -R "${USERNAME}":"${USERNAME}" /usr/local/zed
fi

# If jtop present, give the user access
if [ -S /run/jtop.sock ]; then
  JETSON_STATS_GID="$(stat -c %g /run/jtop.sock)"
  addgroup --gid "${JETSON_STATS_GID}" jtop >/dev/null
  adduser "${USERNAME}" jtop >/dev/null
fi

# Give the user access to I2C (/dev/i2c-*), GPIO (/dev/gpiochip*) and DRM render
# (/dev/dri/render*) devices when present. The nodes are group-owned but the group
# names/GIDs may not match the container's, so resolve each node's GID and add the
# user (creating the group if absent).
for hw_dev in /dev/i2c-* /dev/gpiochip* /dev/dri/render*; do
  [ -e "${hw_dev}" ] || continue
  HW_DEV_GID="$(stat -c %g "${hw_dev}")"
  [ "${HW_DEV_GID}" = "0" ] && continue # root-owned/world-accessible node; no group needed
  HW_DEV_GRP="$(getent group "${HW_DEV_GID}" | cut -d: -f1)"
  if [ -z "${HW_DEV_GRP}" ]; then
    HW_DEV_GRP="hwgrp${HW_DEV_GID}"
    groupadd --gid "${HW_DEV_GID}" "${HW_DEV_GRP}" >/dev/null 2>&1
  fi
  adduser "${USERNAME}" "${HW_DEV_GRP}" >/dev/null 2>&1
done

# Run all entrypoint additions
shopt -s nullglob
for addition in /usr/local/bin/scripts/entrypoint_additions/*.sh; do
  if [[ ${addition} == *".user."* ]]; then
    echo "Running entrypoint extension: ${addition} as user ${USERNAME}"
    gosu "${USERNAME}" "${addition}"
  else
    echo "Sourcing entrypoint extension: ${addition}"
    # shellcheck source=/dev/null
    source "${addition}"
  fi
done

exec gosu "${USERNAME}" "$@"
