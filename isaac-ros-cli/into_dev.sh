#!/usr/bin/env bash
# into_dev.sh -- open a shell in the persistent Isaac ROS dev container (starts it first if stopped),
# as the host user that created it with ./start_dev.sh (not "admin").
# Same `docker exec` that run_dev.py uses for a 2nd `isaac-ros activate` terminal.
# Leaving the shell does NOT stop or delete the container.
#
# Fast path (container already running): ONE `docker inspect` + the final `docker exec`.
# TIMING=1 ./into_dev.sh   prints elapsed milliseconds at each stage.
set -euo pipefail

# default name = isaac_ros_dev_<host user>, same rule as start_dev.sh
NAME="${ISAAC_DEV_CONTAINER:-isaac_ros_dev_${ISAAC_DEV_USER:-$(id -un)}}"

T0="${EPOCHREALTIME/[.,]/}"
tm() { [ "${TIMING:-0}" = "1" ] && echo "[timing] $(( (${EPOCHREALTIME/[.,]/} - T0) / 1000 )) ms  $*" >&2 || true; }

# one call: running?, owner label, host PID, then the container's environment
if ! INFO="$(docker inspect -f '{{.State.Running}}{{"\n"}}{{index .Config.Labels "isaac.dev.user"}}{{"\n"}}{{.State.Pid}}{{"\n"}}{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null)"; then
  echo "[into_dev] container '$NAME' does not exist. Run ./start_dev.sh first." >&2
  if docker inspect isaac_ros_dev_persistent >/dev/null 2>&1; then
    echo "[into_dev] found the container created by the previous script version: isaac_ros_dev_persistent" >&2
    echo "[into_dev]   keep using it:  ISAAC_DEV_CONTAINER=isaac_ros_dev_persistent ./into_dev.sh" >&2
    echo "[into_dev]   or recreate it with the new name/host name:  docker rm -f isaac_ros_dev_persistent && ./start_dev.sh" >&2
  fi
  exit 1
fi
tm "inspect done"
mapfile -t L <<<"$INFO"
RUNNING="${L[0]}"; LABEL="${L[1]}"
WS=""; ENV_USER=""; CUID=""
for line in "${L[@]:3}"; do
  case "$line" in
    ISAAC_ROS_WS=*)   WS="${line#*=}" ;;
    USERNAME=*)       ENV_USER="${line#*=}" ;;
    HOST_USER_UID=*)  CUID="${line#*=}" ;;
  esac
done
CUSER="${ISAAC_DEV_USER:-${LABEL:-${ENV_USER:-$(id -un)}}}"
WS="${WS:-/workspaces/isaac_ros-dev}"
CUID="${CUID:-$(id -u)}"

if [ "$RUNNING" != "true" ]; then
  echo "[into_dev] '$NAME' is stopped, starting it..."
  docker start "$NAME" >/dev/null
  tm "docker start returned"
  # The container runs with --pid=host, so /proc/1 inside it is the HOST's init: watch the container's
  # host-side PID instead, until the entrypoint has handed it to the container user.
  PID="$(docker inspect -f '{{.State.Pid}}' "$NAME")"
  for ((i = 0; i < 600; i++)); do                 # up to ~60 s, polling /proc locally (no docker calls)
    [ "$PID" -gt 0 ] && [ "$(stat -c %u "/proc/$PID" 2>/dev/null || echo x)" = "$CUID" ] && break
    sleep 0.1
  done
  tm "entrypoint finished"
fi

# Let the container user draw on the host's X display even if ~/.Xauthority was regenerated at login
# (same uid inside and outside the container, so this grants nothing new).
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
  xhost "+SI:localuser:$(id -un)" >/dev/null 2>&1 || true
fi

EXEC_ENV=(-e TERM=xterm-256color -e COLORTERM=truecolor -e FORCE_COLOR=true)
[ -n "${DISPLAY:-}" ] && EXEC_ENV+=(-e "DISPLAY=$DISPLAY")

tm "exec into container as $CUSER"
exec docker exec -i -t "${EXEC_ENV[@]}" -u "$CUSER" --workdir "$WS" "$NAME" /bin/bash
