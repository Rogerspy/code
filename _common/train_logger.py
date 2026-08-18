"""标准 AI 训练日志 + 含光 PPU 显卡运行状态(memin/util/power/tflops)。

提供:
- GPUMonitor: 后台线程按 ppu-smi 查询采样物理 PPU 的显存/利用率/功耗/温度,统计峰值。
- measure_flops(fn): 用 torch FlopCounterMode 测一步(forward+backward)的真实 FLOP 数。
- log_step / log_eval / train_summary: 标准(类 PyTorch Lightning / HF Trainer)日志行。

tflops 由 FLOP 计数器测得的"每步 FLOP × 总步数 / 训练时长"得到(ppu-smi 无 FLOPS 字段,
这是标准做法)。
"""
import os
import subprocess
import threading
import time

import torch


def ppu_dev():
    """物理 PPU 索引 = CUDA_VISIBLE_DEVICES(单卡)。"""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return v.split(",")[0] if v else "0"


def query_ppu_now(dev=None):
    """一次 ppu-smi 查询 -> (mem_MiB, util%, power_W, temp_C)。"""
    dev = dev or ppu_dev()
    try:
        out = subprocess.run(
            ["ppu-smi", "--query-ppu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", str(dev)],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        parts = [x.strip() for x in out.split(",")]
        return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except Exception:
        return None, None, None, None


class GPUMonitor:
    """后台线程周期采样 ppu-smi,记录显存/利用率/功耗峰值与均值。"""

    def __init__(self, dev=None, interval=0.5):
        self.dev = dev or ppu_dev()
        self.interval = interval
        self.samples = []  # (mem, util, power, temp)
        self.last = None    # 最近一次后台采样值,供 snapshot_line 用(避免瞬时查询在 kernel 间隙抓到 0)
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            mem, util, pwr, temp = query_ppu_now(self.dev)
            if mem is not None:
                self.samples.append((mem, util, pwr, temp))
                self.last = (mem, util, pwr, temp)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=3)

    def stats(self):
        if not self.samples:
            return {}
        mems = [s[0] for s in self.samples]
        utils = [s[1] for s in self.samples if s[1] is not None]
        pwrs = [s[2] for s in self.samples if s[2] is not None]
        temps = [s[3] for s in self.samples if s[3] is not None]
        return dict(
            peak_mem_mb=max(mems), min_mem_mb=min(mems),
            avg_util=(sum(utils) / len(utils)) if utils else 0.0,
            peak_util=max(utils) if utils else 0.0,
            avg_power=(sum(pwrs) / len(pwrs)) if pwrs else 0.0,
            peak_temp=max(temps) if temps else 0.0,
            n_samples=len(self.samples),
            dev=self.dev,
        )

    def snapshot_line(self, tag="now"):
        # 优先用后台采样器的最近一次样本(反映训练中的真实状态),否则即时查询一次
        if self.last is not None:
            mem, util, pwr, temp = self.last
        else:
            mem, util, pwr, temp = query_ppu_now(self.dev)
        if mem is None:
            return f"[GPU] ppu{self.dev} 采样失败"
        return (f"[GPU] ppu{self.dev} | mem {mem:.0f}MiB/98304MiB | util {util:.0f}% | "
                f"power {pwr:.1f}W | temp {temp:.0f}C")


def measure_flops(fn):
    """在 FlopCounterMode 下运行 fn()(一步 forward+backward),返回该步 FLOP 数(int)。"""
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    try:
        return int(fc.get_total_flops())
    except Exception:
        return int(fc.get_total_flops()) if hasattr(fc, "get_total_flops") else 0


def log_step(epoch, n_epochs, step, n_steps, loss, lr=None, throughput=None, gpu=None):
    """打印一条标准逐 step 训练日志。"""
    parts = [f"epoch {epoch}/{n_epochs}", f"step {step}/{n_steps}", f"loss {loss:.6f}"]
    if lr is not None:
        parts.append(f"lr {lr:.2e}")
    if throughput is not None:
        parts.append(f"{throughput:.2f} samples/s")
    line = "  " + " | ".join(parts)
    if gpu is not None:
        line += "  " + gpu
    print(line)


def log_eval(epoch, acc, n_cases, mean_loss):
    """打印一条 epoch 评估(accuracy)日志。"""
    print(f"  [eval] epoch {epoch} | acc {acc}/{n_cases} ({acc / max(n_cases, 1):.3f}) | mean_loss {mean_loss:.6f}")


def train_summary(model_name, total_steps, train_time, flops_per_step,
                  gpu_stats, final_acc, n_cases, n_epochs):
    """训练结束打印标准汇总(含实测 tflops、显存峰值、利用率、功耗)。"""
    avg_tflops = (flops_per_step * total_steps / train_time) if (flops_per_step and train_time > 0) else 0.0
    gs = gpu_stats or {}
    peak_mem = gs.get("peak_mem_mb", 0.0)
    avg_util = gs.get("avg_util", 0.0)
    peak_util = gs.get("peak_util", 0.0)
    avg_power = gs.get("avg_power", 0.0)
    peak_temp = gs.get("peak_temp", 0.0)
    samples = gs.get("n_samples", 0)
    dev = gs.get("dev", ppu_dev())
    tput = (n_cases * n_epochs / train_time) if train_time > 0 else 0.0
    bar = "=" * 76
    print(bar)
    print(f"TRAIN SUMMARY | {model_name} | device PPU-ZW810E (cuda:0, physical ppu{dev})")
    print(f"  steps={total_steps}  epochs={n_epochs}  train_time={train_time:.3f}s  throughput={tput:.2f} samples/s")
    print(f"  final_acc={final_acc}/{n_cases} ({final_acc / max(n_cases, 1):.3f})")
    print(f"  GPU mem: peak {peak_mem:.0f}MiB / 98304MiB   (ppu-smi, {samples} samples)")
    print(f"  GPU util: avg {avg_util:.1f}%  peak {peak_util:.1f}%   power avg {avg_power:.1f}W   temp peak {peak_temp:.0f}C")
    print(f"  TFLOPS (measured: FLOP_counter {flops_per_step/1e6:.2f}M/step × {total_steps} steps / {train_time:.3f}s): {avg_tflops/1e12:.3f} TFLOPS")
    print(bar)
    return dict(avg_tflops=avg_tflops, peak_mem_mb=peak_mem, avg_util=avg_util, peak_util=peak_util)
