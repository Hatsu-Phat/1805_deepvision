"""Server environment check before any GPU job (Phase 2 smoke test 1).

Checks: torch sees the GPU and was built for its architecture, bf16 and GPU JPEG decoding work,
how much of the shared GPU is already in use, the container limits (cgroup RAM / CPU), free disk
and the size of our own ~/deepvision_* directories against the budget, and that the rclone remote
is reachable. Prints one PASS / WARN / FAIL line per check; exits with 1 if anything FAILs.

    python -m deepvision.envcheck --remote gdrive:DeepVision/dataset_v1 --out ~/deepvision_runs/env_check.txt
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

OWN_DIRS = ("deepvision_runs", "deepvision_features", "deepvision_cache")


def _run(args: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def cgroup_limits() -> dict:
    out = {}
    for name in ("memory.max", "memory.peak", "memory.current", "cpu.max"):
        p = Path("/sys/fs/cgroup") / name
        if p.exists():
            out[name] = p.read_text().strip()
    return out


def check_torch(results: list) -> None:
    try:
        import torch
    except ImportError:
        results.append(("FAIL", "torch", "not installed"))
        return
    results.append(("PASS", "torch", f"{torch.__version__} cuda {torch.version.cuda}"))
    for mod in ("torchvision", "timm", "transformers", "sklearn"):
        try:
            results.append(("PASS", mod, importlib.import_module(mod).__version__))
        except ImportError:
            results.append(("WARN", mod, "not installed"))
    if not torch.cuda.is_available():
        results.append(("FAIL", "cuda", "torch.cuda.is_available() is False"))
        return
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    props = torch.cuda.get_device_properties(0)
    ok = arch in torch.cuda.get_arch_list()
    results.append(("PASS" if ok else "FAIL", "gpu arch",
                    f"{props.name} {arch}, torch arch list {' '.join(torch.cuda.get_arch_list())}"))
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        float((a @ a).float().sum())
        results.append(("PASS", "bf16", "matmul on GPU OK"))
    except Exception as e:  # noqa: BLE001 - report any CUDA failure
        results.append(("FAIL", "bf16", repr(e)))
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**20
    results.append(("PASS" if used < 1024 else "WARN", "vram",
                    f"free {free / 2**20:.0f} / {total / 2**20:.0f} MiB (in use by others: {used:.0f} MiB)"))
    try:
        from torchvision.io import decode_jpeg, encode_jpeg

        img = (torch.rand(3, 224, 224) * 255).to(torch.uint8)
        out = decode_jpeg([encode_jpeg(img)] * 4, device="cuda")
        results.append(("PASS", "gpu jpeg decode", f"{len(out)} x {tuple(out[0].shape)} on {out[0].device}"))
    except Exception as e:  # noqa: BLE001
        results.append(("WARN", "gpu jpeg decode", repr(e)))


def check_system(results: list, home: Path, budget_gib: float) -> None:
    code, out = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader"])
    results.append(("PASS" if code == 0 else "WARN", "nvidia-smi", out.splitlines()[0] if out else "not found"))
    code, out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    if code == 0:
        results.append(("PASS" if not out else "WARN", "gpu processes", out.replace("\n", "; ") or "none visible"))
    limits = cgroup_limits()
    if "memory.max" in limits:
        mem = limits["memory.max"]
        mem_text = mem if mem == "max" else f"{int(mem) / 2**30:.2f} GiB"
        results.append(("PASS", "cgroup", f"memory.max {mem_text}, cpu.max {limits.get('cpu.max', '?')}"))
    results.append(("PASS", "cpus", f"os.cpu_count {os.cpu_count()}, usable {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'}"))
    du = shutil.disk_usage(home)
    results.append(("PASS" if du.free > 100 * 2**30 else "WARN", "disk",
                    f"{home}: free {du.free / 2**30:.0f} / {du.total / 2**30:.0f} GiB"))
    own = sum(dir_size(home / d) for d in OWN_DIRS if (home / d).exists()) / 2**30
    results.append(("PASS" if own < budget_gib else "FAIL", "own dirs",
                    f"{' + '.join(OWN_DIRS)} = {own:.2f} GiB (budget {budget_gib:.0f} GiB)"))


def check_rclone(results: list, remote: str, rclone: str) -> None:
    binary = shutil.which(rclone)
    if not binary:
        results.append(("FAIL", "rclone", "not found"))
        return
    code, out = _run([binary, "version"])
    results.append(("PASS", "rclone", out.splitlines()[0] if out else "?"))
    code, out = _run([binary, "lsf", remote.rstrip("/") + "/shards", "--max-depth", "1"])
    results.append(("PASS" if code == 0 else "FAIL", "remote", f"{remote}/shards: " + (out.replace("\n", " ") if code == 0 else out[-300:])))
    code, out = _run([binary, "about", remote.split(":", 1)[0] + ":"])
    results.append(("PASS" if code == 0 else "WARN", "remote space", out.replace("\n", "; ")[-300:]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", default="gdrive:DeepVision/dataset_v1")
    ap.add_argument("--rclone", default="rclone")
    ap.add_argument("--home", default="~")
    ap.add_argument("--budget-gib", type=float, default=80.0)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)
    results: list[tuple[str, str, str]] = [("PASS", "python", sys.version.split()[0])]
    check_torch(results)
    check_system(results, Path(os.path.expanduser(args.home)), args.budget_gib)
    check_rclone(results, args.remote, args.rclone)
    text = "\n".join(f"{status:4s}  {name:16s} {detail}" for status, name, detail in results)
    print(text)
    if args.out:
        out = Path(os.path.expanduser(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    sys.exit(1 if any(s == "FAIL" for s, _, _ in results) else 0)


if __name__ == "__main__":
    main()
