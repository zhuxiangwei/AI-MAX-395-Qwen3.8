#!/bin/bash
# rocm-wait-devices.sh - 等待 ROCm 设备节点就绪后再返回
#
# 背景（2026-08-30）：
#   开机后 /dev/kfd 约 +3s、/dev/dri/renderD128 约 +6s 才由 udev 创建，
#   但 user 级 systemd 单元只依赖 network.target 时，服务开机后约 4s 就 exec
#   python，HIP 初始化时设备节点不存在 -> ROCm 内部 abort() -> SIGABRT。
#   user 级 systemd 不支持 device unit（After=dev-kfd.device 无效），
#   因此只能在 ExecStartPre 里显式轮询。
#
# 用法：rocm-wait-devices.sh [超时秒数，默认 90]
#   超时后仍返回 0（放行），只打警告 —— 依赖 Restart=on-failure 兜底，
#   避免设备节点永远不出现时服务彻底起不来。

set -u
TIMEOUT="${1:-90}"

dev_ok() {
    # 1) 节点存在  2) 当前用户可读写（udev 创建后 chown/chmod 有窗口期）
    [ -c /dev/kfd ] && [ -r /dev/kfd ] && [ -w /dev/kfd ] || return 1
    local r
    for r in /dev/dri/renderD*; do
        [ -c "$r" ] && [ -r "$r" ] && [ -w "$r" ] && return 0
    done
    return 1
}

if dev_ok; then
    exit 0
fi

echo "rocm-wait-devices: waiting for /dev/kfd + /dev/dri/renderD* (timeout ${TIMEOUT}s)..." >&2
for i in $(seq 1 "$TIMEOUT"); do
    sleep 1
    if dev_ok; then
        echo "rocm-wait-devices: ready after ${i}s" >&2
        exit 0
    fi
done

echo "rocm-wait-devices: WARN devices not ready after ${TIMEOUT}s, starting anyway" >&2
exit 0
