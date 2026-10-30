#!/usr/bin/env bash
# start_dev.sh -- create (once) a PERSISTENT Isaac ROS dev container, running as the HOST USER that launches this script.
#
# Mirrors what /usr/lib/isaac-ros-cli/run_dev.py runs for `isaac-ros activate` (aarch64/Jetson path), with these differences:
#   1. no `--rm`   -> container is NOT deleted when you leave it (apt installs etc. are kept)
#   2. `-d`        -> runs in the background; attach with ./into_dev.sh
#   3. the in-container user is YOU (e.g. "chery"), not "admin": the image's entrypoint reads $USERNAME
#      (Dockerfile.isaac_ros: ENV USERNAME=admin) so we override it with `-e USERNAME=<you>`; all
#      /home/admin/... mount targets become /home/<you>/...
#   4. a per-user container name: isaac_ros_dev_<user>  (e.g. isaac_ros_dev_chery)
#   5. the in-container host name is "docker" (prompt: chery@docker:/workspaces/isaac_ros-dev$) instead of the
#      host's name; with --network host Docker would otherwise copy the host's name ("localhost").
#      It is fixed when the container is CREATED -- changing it later means `docker rm -f <name>` and re-creating.
#
# WARNING: never name it isaac_ros_dev_container*.  run_dev.py runs `docker rm` on any EXITED container whose
# name matches that string each time you run `isaac-ros activate`.  (So a host user literally called "container"
# must set ISAAC_DEV_CONTAINER to something else.)
#
# Extra mounts are read from .isaac_ros_dev-dockerargs, searched like run_dev.py does:
#   $DOCKER_ARGS_FILE, ~/.isaac_ros_dev-dockerargs, then <this dir>/.isaac_ros_dev-dockerargs
#   (or /etc/isaac-ros-cli/.isaac_ros_dev-dockerargs).   "/home/admin/" in those files is rewritten to "/home/<you>/".
#   Mounts whose host path does not exist are skipped (so Docker doesn't create root-owned dirs in your home).
#
# Usage:  ./start_dev.sh             create it (or start it if it exists but is stopped)
#         DRY_RUN=1 ./start_dev.sh   only print the docker command
# Env:    ISAAC_DEV_CONTAINER (default isaac_ros_dev_<user>), ISAAC_DEV_HOSTNAME (default docker),
#         ISAAC_DEV_IMAGE, ISAAC_DEV_USER (override the user name), ISAAC_ROS_WS, TIMING=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# ---- helpers ---------------------------------------------------------------------------------------
# TIMING=1 ./start_dev.sh   prints elapsed milliseconds at each stage
T0="${EPOCHREALTIME/[.,]/}"
tm() { [ "${TIMING:-0}" = "1" ] && echo "[timing] $(( (${EPOCHREALTIME/[.,]/} - T0) / 1000 )) ms  $*" >&2 || true; }

# Wait until the container's main process has been handed to the container user by the entrypoint
# (user/groups created, sudoers written ...). NOTE: the container runs with --pid=host, so /proc/1
# inside it is the HOST's init -- we must look at the container's *host* PID from the host side.
wait_ready() {  # $1 = container name, $2 = uid of the container user
  local pid i
  pid="$(docker inspect -f '{{.State.Pid}}' "$1" 2>/dev/null || echo 0)"
  for ((i = 0; i < 600; i++)); do             # up to ~60 s, polling /proc locally every 0.1 s (no docker calls)
    [ "$pid" -gt 0 ] && [ "$(stat -c %u "/proc/$pid" 2>/dev/null || echo x)" = "$2" ] && return 0
    sleep 0.1
  done
  echo "[start_dev] warning: entrypoint did not finish within 60 s; continuing anyway" >&2
  return 1
}

# ---- who runs the container ---------------------------------------------------------------------------
CUSER="${ISAAC_DEV_USER:-$(id -un)}"
CUID="$(id -u)"; CGID="$(id -g)"
CHOME="/home/$CUSER"
if [ "$CUID" = "0" ] || [ "$CUSER" = "root" ]; then
  echo "[start_dev] don't run this as root / with sudo: the container user must be your normal user." >&2; exit 1
fi
if ! [[ "$CUSER" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
  echo "[start_dev] '$CUSER' is not a valid Linux user name for the container (use ISAAC_DEV_USER=<name>)." >&2; exit 1
fi

NAME="${ISAAC_DEV_CONTAINER:-isaac_ros_dev_${CUSER}}"
case "$NAME" in
  *isaac_ros_dev_container*) echo "[start_dev] name '$NAME' must not contain 'isaac_ros_dev_container' (see comment at top); set ISAAC_DEV_CONTAINER=<other name>" >&2; exit 1 ;;
esac
if ! [[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "[start_dev] '$NAME' is not a valid Docker container name." >&2; exit 1
fi

HNAME="${ISAAC_DEV_HOSTNAME:-docker}"
if ! [[ "$HNAME" =~ ^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$ ]]; then
  echo "[start_dev] '$HNAME' is not a valid host name (use ISAAC_DEV_HOSTNAME=<name>)." >&2; exit 1
fi

IMAGE="${ISAAC_DEV_IMAGE:-nvcr.io/nvidia/isaac/ros:isaac_ros-realsense_1aab99481814f8622a5fdd169073390a-arm64-jetpack}"
ISAAC_DIR="$(realpath -m "${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}")"
PLATFORM_ID="arm64-jetpack"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ]; then
  if INFO="$(docker inspect -f '{{.State.Running}}{{"\n"}}{{index .Config.Labels "isaac.dev.user"}}{{"\n"}}{{.Config.Hostname}}' "$NAME" 2>/dev/null)"; then
    tm "inspect done"
    mapfile -t L <<<"$INFO"
    RUNNING="${L[0]}"; OWNER="${L[1]:-}"; OLD_HOST="${L[2]:-}"
    [ -n "$OWNER" ] && [ "$OWNER" != "$CUSER" ] && \
      echo "[start_dev] note: '$NAME' was created for user '$OWNER'. To switch user: docker rm -f $NAME (installed packages in it are lost)." >&2
    [ -n "$OLD_HOST" ] && [ "$OLD_HOST" != "$HNAME" ] && \
      echo "[start_dev] note: '$NAME' has host name '$OLD_HOST' (wanted '$HNAME'). It can only be set at creation: docker rm -f $NAME, then ./start_dev.sh." >&2
    if [ "$RUNNING" = "true" ]; then
      echo "[start_dev] '$NAME' is already running.  ->  ./into_dev.sh"
    else
      echo "[start_dev] '$NAME' exists but is stopped; starting it (installed packages are kept)."
      docker start "$NAME" >/dev/null
      tm "docker start returned"
      wait_ready "$NAME" "$CUID" || true
      tm "entrypoint finished"
      echo "[start_dev] ready.  ->  ./into_dev.sh"
    fi
    exit 0
  fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[start_dev] image not found: $IMAGE" >&2
    if docker image inspect cached_isaac_run_dev_image_local:latest >/dev/null 2>&1; then
      echo "[start_dev] falling back to cached_isaac_run_dev_image_local:latest" >&2
      IMAGE="cached_isaac_run_dev_image_local:latest"
    else
      echo "[start_dev] run 'docker images' and set ISAAC_DEV_IMAGE=<image> ./start_dev.sh" >&2
      exit 1
    fi
  fi
fi

ARGS=()
# add "-v src:dst[:opts]" only if the host path exists
mount_if_exists() { [ -e "$1" ] && ARGS+=(-v "$1:$2${3:+:$3}") || true; }

ARGS+=(-i -t -d
  --privileged --network host --ipc=host
  --label "isaac.dev.user=$CUSER"
  --hostname "$HNAME" --add-host "$HNAME:127.0.0.1"
  -e "USERNAME=$CUSER"
  -e TERM=xterm-256color -e COLORTERM=truecolor -e FORCE_COLOR=true
  --workdir /workspaces/isaac_ros-dev
  -e "ISAAC_ROS_PLATFORM=$PLATFORM_ID")

# --- same as run_dev.py: get_docker_args() ---------------------------------------------------------
ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
mount_if_exists "$HOME/.Xauthority" "$CHOME/.Xauthority" rw
for f in .bashrc .bash_profile .profile .dircolors; do          # host shell config, read-only
  [ -f "$HOME/$f" ] && ARGS+=(-v "$HOME/$f:$CHOME/$f:ro")
done
ARGS+=(-e "DISPLAY=${DISPLAY:-:0}"
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
  -e ROS_DOMAIN_ID -e USER
  -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev
  -e "HOST_USER_UID=$CUID" -e "HOST_USER_GID=$CGID")
[ -e /etc/fastos-release ] && ARGS+=(-v /etc/fastos-release:/etc/fastos-release:ro)

# Jetson-specific mounts (aarch64 branch of run_dev.py)
mount_if_exists /usr/bin/tegrastats /usr/bin/tegrastats
mount_if_exists /sys/kernel/debug /sys/kernel/debug ro
ARGS+=(-v /tmp/:/tmp/)
mount_if_exists /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu/tegra
mount_if_exists /usr/src/jetson_multimedia_api /usr/src/jetson_multimedia_api
mount_if_exists /usr/src/jetson_sipl_api /usr/src/jetson_sipl_api
ARGS+=(--pid=host)
mount_if_exists /usr/share/vpi3 /usr/share/vpi3
mount_if_exists /dev/input /dev/input
mount_if_exists /dev/bus/usb /dev/bus/usb                       # RealSense USB access
shopt -s nullglob
for d in /dev/coe-chan-*; do ARGS+=(--device="$d"); done
shopt -u nullglob
if jtop_gid="$(getent group jtop | cut -d: -f3)" && [ -n "$jtop_gid" ]; then
  mount_if_exists /run/jtop.sock /run/jtop.sock ro
  ARGS+=(--group-add "$jtop_gid")
fi
# (the CLI also forwards SSH_AUTH_SOCK; omitted on purpose: the socket path changes at every login)

# --- extra args from .isaac_ros_dev-dockerargs (same search order as run_dev.py) --------------------
add_dockerargs_file() {
  local file="$1" line re src rest p
  re='^-v[[:space:]]+`realpath[[:space:]]+([^`]+)`:(.*)$'
  echo "[start_dev] extra docker args from $file" >&2
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
    line="${line//\/home\/admin\//$CHOME/}"                       # admin -> current user
    if [[ "$line" =~ $re ]]; then                                  # -v `realpath ~/x`:/dst[:opts]
      p="${BASH_REMATCH[1]}"; rest="${BASH_REMATCH[2]}"; p="${p/#\~/$HOME}"
      src="$(realpath -m "$p")"
      if [ -e "$src" ]; then ARGS+=(-v "$src:$rest"); else echo "[start_dev]   skip (missing): $src" >&2; fi
    elif [[ "$line" == "-v "* ]]; then                             # plain -v src:dst[:opts]
      rest="${line#-v }"; src="${rest%%:*}"; src="${src/#\~/$HOME}"
      if [ -e "$src" ]; then ARGS+=(-v "$src:${rest#*:}"); else echo "[start_dev]   skip (missing): $src" >&2; fi
    else
      read -r -a words <<<"$line"; ARGS+=("${words[@]}")
    fi
  done < "$file"
}
for f in "${DOCKER_ARGS_FILE:-}" "$HOME/.isaac_ros_dev-dockerargs"; do
  [ -n "$f" ] && [ -f "$f" ] && add_dockerargs_file "$f"
done
if   [ -f "$SCRIPT_DIR/.isaac_ros_dev-dockerargs" ];            then add_dockerargs_file "$SCRIPT_DIR/.isaac_ros_dev-dockerargs"
elif [ -f /etc/isaac-ros-cli/.isaac_ros_dev-dockerargs ];       then add_dockerargs_file /etc/isaac-ros-cli/.isaac_ros_dev-dockerargs
fi

# --- workspace, time zone, GPU, entrypoint ------------------------------------------------------------
ARGS+=(-v "$ISAAC_DIR:/workspaces/isaac_ros-dev"
  -v /etc/localtime:/etc/localtime:ro
  --name "$NAME"
  --gpus all
  --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh
  "$IMAGE" /bin/bash)

if [ "$DRY_RUN" = "1" ]; then
  printf 'docker run'; printf ' %q' "${ARGS[@]}"; echo
  exit 0
fi

echo "[start_dev] creating '$NAME' (user: $CUSER, uid=$CUID gid=$CGID, host name: $HNAME) from $IMAGE"
docker run "${ARGS[@]}" >/dev/null

tm "docker run returned"
wait_ready "$NAME" "$CUID" || true
tm "entrypoint finished"
echo "[start_dev] ready.  ->  ./into_dev.sh"
