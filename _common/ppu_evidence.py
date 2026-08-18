"""PPU training-evidence logger (含光 PPU-ZW810E via CUDA-compat).

打印不可辩驳的证据,证明计算发生在本机含光 PPU 上,而非 NVIDIA 显卡或 CPU。
用法(在 train.py / infer.py 中):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
    from ppu_evidence import header, assert_on_cuda, mem, smi
    header("<model>")                 # 证明环境是 PPU(非 nvidia、非 cpu)
    assert_on_cuda(model, "<model>")   # 证明模型参数在 cuda:0(PPU)
    mem("after_train")                # PPU 显存占用
    smi("snapshot")                   # ppu-smi 真实显存
"""
import os
import subprocess

import torch


def header(tag: str = "model"):
    """打印并断言:torch.cuda 可用 且 设备名含 PPU(本机含光卡)。"""
    print("=" * 66)
    avail = torch.cuda.is_available()
    cnt = torch.cuda.device_count() if avail else 0
    name = torch.cuda.get_device_name(0) if avail else "<none>"
    print(f"[PPU-EVIDENCE] {tag}")
    print(f"[PPU-EVIDENCE] torch={torch.__version__}")
    print(f"[PPU-EVIDENCE] torch.cuda.is_available()={avail}")
    print(f"[PPU-EVIDENCE] device_count={cnt}")
    print(f"[PPU-EVIDENCE] device(0).name={name!r}  <- 须为 'PPU-ZW810E'(本机含光卡)")
    print(f"[PPU-EVIDENCE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"[PPU-EVIDENCE] PJRT_DEVICE={os.environ.get('PJRT_DEVICE')!r}")
    assert avail, "FATAL: cuda 不可用 -> 会退回 CPU,不是 PPU"
    assert "PPU" in name, f"FATAL: 设备名 {name!r} 不含 PPU -> 不是本机含光卡(可能 nvidia)"
    print(f"[PPU-EVIDENCE] PASS: 设备为本机含光 PPU(非 NVIDIA、非 CPU)")
    print("=" * 66)


def assert_on_cuda(model, tag: str = "model"):
    """打印并断言:模型参数在 cuda:0 且 PPU 显存>0。"""
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = None
    alloc = torch.cuda.memory_allocated() / 1024**2
    print(f"[PPU-EVIDENCE] {tag} first_param.device={dev}  cuda_mem_allocated={alloc:.1f}MB")
    assert dev is not None and dev.type == "cuda", f"FATAL: {tag} 参数在 {dev},非 cuda -> CPU 退化!"
    assert alloc > 0, f"FATAL: {tag} cuda 显存=0 -> 模型未上 PPU"
    return dev


def mem(tag: str = ""):
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    print(f"[PPU-EVIDENCE] mem{' ' + tag if tag else ''}: allocated={a:.1f}MB reserved={r:.1f}MB")


def smi(tag: str = ""):
    """采样 ppu-smi,打印 PPU 真实显存占用(本机监控工具,非 nvidia-smi)。"""
    try:
        out = subprocess.run(
            ["ppu-smi"], capture_output=True, text=True, timeout=20
        ).stdout
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        print(f"[ppu-smi] {'snapshot ' + tag if tag else ''}({len(lines)} lines)")
        for ln in lines[:40]:
            print(f"[ppu-smi] {ln}")
    except Exception as e:
        print(f"[ppu-smi] 不可用: {e!r}")
