# Setting Up Local YOLO Inference on the Jetson Orin Nano

This guide walks through installing everything needed to run YOLO on the
Jetson's GPU and building a TensorRT engine — without breaking this
repository's pinned `numpy` and `opencv-contrib-python` versions.

**This guide works for YOLOv8, YOLOv11, YOLO26, and any other model
variant trained through Roboflow or ultralytics.** The `ultralytics` pip
package is the single unified runtime for all of these — it auto-detects
the architecture from your `.pt` weights file. The installation and
engine export steps are identical regardless of which YOLO version you
trained with.

**Target environment** (all course Jetsons are identical):
- Jetson Orin Nano, JetPack 6 (L4T R36.4.7)
- CUDA 12.6, TensorRT 10.3, Python 3.10.12

The install ordering matters. **Read each step before running the
commands in it.** The single biggest way to break this repo is to let
`pip` upgrade `numpy` or install `opencv-python` on top of
`opencv-contrib-python`.

> **TL;DR of the gotcha:** plain `pip install ultralytics` will pull
> down CPU-only PyTorch + `opencv-python` and may upgrade `numpy` past
> `1.26`. Every one of those breaks something:
>   * CPU PyTorch → no GPU inference.
>   * `opencv-python` alongside `opencv-contrib-python` → broken `cv2.aruco`.
>   * `numpy >= 2.0` → ABI mismatch with `cv_bridge`, `scipy`, OpenCV.
>
> Always use `--no-deps` with `ultralytics` and install its real
> dependencies manually, skipping the four that we already have pinned
> (`torch`, `torchvision`, `numpy`, `opencv-python`).

---

## Step 0 — Verify your system

Confirm your Jetson matches the expected environment before proceeding.

```bash
cat /etc/nv_tegra_release
# Expected: R36 (release), REVISION: 4.7 ...

/usr/local/cuda/bin/nvcc --version
# Expected: Cuda compilation tools, release 12.6 ...

python3 --version
# Expected: Python 3.10.12

tegrastats --interval 1000
# Ctrl-C after a couple of lines — confirm you see GR3D_FREQ in the output
# (this proves the GPU is visible)
```

If your JetPack version or Python version differs, stop here and
re-flash the Jetson to the course image before continuing.

If `nvcc` is not found at all, CUDA is not on your `PATH`. Fix it:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # should now show release 12.6
```

If `nvcc` reports a CUDA version other than 12.6, your JetPack install
is non-standard. The torch and onnxruntime-gpu wheels in this guide are
built for CUDA 12.6 — a version mismatch will cause silent failures or
crashes at inference time. Re-flash to the course image.

---

## Step 1 — Install the repo's pinned baseline

Start from a clean baseline with this repo's pinned `numpy` and
`opencv-contrib-python` already in place.

```bash
cd <your-checkout-of-this-repo>

pip install -r Project5_WS/requirements.txt
```

Verify:

```bash
python3 -c "import numpy; print('numpy', numpy.__version__)"   # → 1.26.x
python3 -c "import cv2; print('cv2', cv2.__version__)"         # → 4.10.x
python3 -c "import cv2; print('aruco' in dir(cv2))"            # → True (contrib)
```

If any of those fail, fix them before moving on. YOLO on top of a broken
baseline will burn hours.

---

## Step 2 — Install Jetson-native PyTorch and torchvision

**Do not use `pip install torch`.** On `aarch64` that installs a
CPU-only PyTorch wheel from PyPI — inference will run, but on the CPU,
which defeats the whole point.

The [jetson-ai-lab](https://www.jetson-ai-lab.com) project (maintained by
NVIDIA) hosts Jetson-specific CUDA-enabled wheels. For our JetPack 6 /
CUDA 12.6 environment:

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.8.0 torchvision==0.23.0
```

> **Why pin `torch==2.8.0`?**
>
> The jetson-ai-lab index also carries torch 2.10+ wheels, but those are
> linked against NVIDIA's cuDSS (CUDA Direct Sparse Solver) library,
> which JetPack 6 does not ship and is not in any default apt repo.
> `import torch` will fail with:
>
> ```
> ImportError: libcudss.so.0: cannot open shared object file
> ```
>
> The pip-installable `nvidia-cudss-cu12` package "works", but it drops
> its `.so` files in a non-standard layout (`nvidia/cu12/lib/`) that
> torch's runtime loader does not search. The only workaround is a
> fragile `LD_LIBRARY_PATH` shim that breaks any process not launched
> from your configured shell.
>
> Torch 2.8.0 predates the cuDSS dependency — use it and avoid the
> entire mess. `torchvision==0.23.0` is the matching version.

### Verify CUDA is visible

```bash
python3 -c "import torch; print('torch', torch.__version__, \
    '| cuda', torch.cuda.is_available(), \
    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected: `torch 2.8.0 | cuda True | Orin`. If it says `cuda False`,
you installed the CPU wheel — `pip uninstall torch torchvision` and
rerun the install command above.

### Re-check the pinned stack

The torch install can occasionally disturb numpy. Check:

```bash
python3 -c "import numpy; print('numpy', numpy.__version__)"   # → still 1.26.x
```

If numpy has moved: `pip install --force-reinstall 'numpy==1.26.*'` and
verify again before continuing.

---

## Step 3 — Install `ultralytics` (no-deps) and its dependencies manually

This is the critical step. `ultralytics` declares `numpy`,
`opencv-python`, `torch`, and `torchvision` as dependencies — we already
have pinned or GPU-specialized versions of all four and must not let
`pip` replace them.

```bash
pip install --no-deps ultralytics
```

Then install ultralytics' remaining dependencies **by hand**, skipping
the four we already have:

```bash
pip install \
    matplotlib \
    pillow \
    tqdm \
    psutil \
    py-cpuinfo \
    pandas \
    seaborn \
    requests \
    ultralytics-thop
```

### Export-time dependencies

The TensorRT engine export (Step 5) converts `.pt → .onnx → .engine`.
Ultralytics needs `onnx`, `onnxslim`, and `onnxruntime-gpu` for this
pipeline. **You must install these manually** — ultralytics will try to
auto-install them during export, but the auto-install fails because the
`onnxruntime-gpu` package has no `aarch64` wheel on PyPI. Use the
jetson-ai-lab index for that wheel:

```bash
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    onnx onnxslim onnxruntime-gpu
```

`--extra-index-url` (not `--index-url`) means pip tries PyPI first and
falls back to the Jetson mirror for wheels PyPI doesn't carry.

### Verify the full stack

```bash
python3 -c "
import numpy, cv2, torch
from ultralytics import YOLO
print('numpy', numpy.__version__)          # → 1.26.x
print('cv2  ', cv2.__version__)            # → 4.10.x
print('aruco', 'aruco' in dir(cv2))        # → True
print('cuda ', torch.cuda.is_available())  # → True
print('ultralytics ok')
"
```

All five checks must pass. If any regressed, reinstall the offender:

```bash
pip install --force-reinstall 'numpy==1.26.*'
pip install --force-reinstall 'opencv-contrib-python==4.10.*'
pip uninstall -y opencv-python   # if it snuck in — conflicts with contrib
```

> **Note:** `pip` may report that `ultralytics` is missing `opencv-python` — this is expected and harmless; `opencv-contrib-python` satisfies the same need. After running any reinstalls above, re-run the verification block.

---

## Step 4 — Smoke-test GPU inference

A quick sanity test before the long TensorRT export:

```bash
python3 - <<'PY'
import torch, time
from ultralytics import YOLO

model = YOLO('Project6_Class_Distro/src/vision_processing_package/models/your_vision_model_here.pt')
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')

dummy = torch.zeros(1, 3, 640, 640, device='cuda').half()
model.predict(dummy, device=0, half=True, verbose=False)        # warm-up

t0 = time.perf_counter()
for _ in range(10):
    model.predict(dummy, device=0, half=True, verbose=False)
avg_ms = (time.perf_counter() - t0) * 100                       # sec*1000/10
print(f'{avg_ms:.1f} ms / inference (avg of 10)')
PY
```

> **Run this from the repo root** (the parent of `Project6_Class_Distro/`).

Expect roughly **20–60 ms** per inference for `yolov8n` at 640×640 in
FP16. The TensorRT engine in the next step will improve the raw
forward-pass time, but overall end-to-end latency also includes
preprocessing, postprocessing, and tracker overhead — so real pipeline
frame times will be higher than the raw forward pass (see
"Understanding inference times" below).

---

## Step 5 — Export the TensorRT FP16 engine

One-shot conversion of your custom-trained `your_vision_model_here.pt` into a
Jetson-specific `.engine` file. This runs the `.pt → .onnx → .engine`
pipeline.

### Before you start

- **Close other GPU workloads.** The TensorRT builder profiles many
  kernel variants per layer and needs GPU memory headroom. On Orin Nano
  (8 GB shared CPU+GPU), a concurrent `colcon build` or other heavy
  process can push the builder into memory pressure. It will still
  usually succeed, but with sub-optimal kernel choices (see
  Troubleshooting).

- **Pin clocks for faster + more reliable builds** (optional but
  recommended):
  ```bash
  sudo jetson_clocks
  ```

### Run the export

From the repo root:

```bash
python3 Project6_Class_Distro/src/vision_processing_package/models/pt_to_engine.py
```

The script defaults to `models/your_vision_model_here.pt`. To export a different weights
file (e.g. yolov8n.pt):

```bash
python3 Project6_Class_Distro/src/vision_processing_package/models/pt_to_engine.py \
    Project6_Class_Distro/src/vision_processing_package/models/yolov8n.pt
```

Expect **3–10 minutes** on Orin Nano. The output will look like:

```
Loading .../your_vision_model_here.pt ...
Exporting to TensorRT engine (imgsz=640, device=0, half=True) ...
ONNX: export success ✅ ...
TensorRT: starting export with TensorRT 10.3.0 ...
TensorRT: building FP16 engine as .../your_vision_model_here.engine
...
TensorRT: export success ✅ ...
Wrote: .../your_vision_model_here.engine
```

The resulting `your_vision_model_here.engine` will be written next to the original
`your_vision_model_here.pt` in the `models/` directory.

### What the warnings during export mean

During the build you will likely see two types of warnings that look
alarming but are **not fatal**:

- **`NvMapMemAllocInternalTagged: ... error 12`** — Jetson's DMA-BUF
  allocator is being asked for more memory than is free. TensorRT
  silently retries at a smaller size and continues. Usually caused by
  other workloads competing for the shared 8 GB. The build still
  succeeds.

- **`[TRT] [E] Error Code: 9: Skipping tactic ... Cask Gemm execution`**
  — TensorRT profiles multiple GEMM algorithm variants per layer and
  picks the fastest. "Skipping tactic" means one variant couldn't
  allocate its workspace, so TRT falls back to the next candidate. The
  engine still builds, just with a slightly smaller tactic search space.
  If you want the most optimal engine, re-export after stopping
  competing workloads and running `sudo jetson_clocks`.

### Re-export after JetPack upgrades

TensorRT engines are bound to the TRT version and GPU architecture. An
engine built on one JetPack version will not load on a different one.
Always re-export on the target Jetson after any JetPack upgrade.

---

## Understanding inference times

If your TensorRT engine doesn't seem dramatically faster than the raw
`.pt` weights, that's expected. Here's why:

The **TensorRT forward pass** (the GPU inference itself) is typically
2–3× faster than raw PyTorch. But any end-to-end measurement that wraps
ultralytics' `.predict()` or `.track()` call includes additional stages
that TensorRT does not accelerate:

| Stage | Runs on | Typical time (Orin Nano) |
|---|---|---|
| Preprocessing (resize, normalize, CPU→GPU) | CPU | 5–10 ms |
| **TensorRT forward pass** | **GPU** | **10–15 ms** (the part TRT accelerates) |
| Postprocessing (NMS, box decode) | CPU | ~5 ms |
| Tracker (ByteTrack / BoT-SORT) | CPU | 5–15 ms |
| Python + message overhead | CPU | ~5 ms |

So a total of **30–50 ms** per frame is normal even with a well-built
TensorRT engine.

### Performance tuning tips

- **`sudo jetson_clocks`** — pins CPU/GPU/memory clocks at their
  maximum within the current power mode. This is a free performance gain
  with no power-envelope change. Safe on battery — it doesn't raise the
  power cap, just prevents clock throttling.

- **Check `nvpmodel`** — `sudo nvpmodel -q` shows your current power
  mode. Higher modes allow higher clocks and more active CPU cores.
  `sudo nvpmodel -m 0` is MAXN (uncapped) but draws more power — avoid
  on battery.

- **Rebuild the engine without memory pressure** — if your first export
  showed many "Skipping tactic" warnings, a rebuild on a quiet system
  (nothing else running, `jetson_clocks` active) may shave 5–10% off
  inference time.

- **Smaller model** — `yolov8n` (nano) is the lightest. `yolov8s`
  (small) is ~2× heavier. If inference time is the bottleneck, use the
  smallest model whose accuracy meets your needs.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available() == False` | Installed the CPU wheel from PyPI. `pip uninstall torch torchvision` then redo Step 2. |
| `ImportError: libcudss.so.0: cannot open shared object file` | You installed `torch >= 2.10`. JetPack 6 doesn't ship cuDSS. `pip uninstall -y torch torchvision` and reinstall with the `torch==2.8.0 torchvision==0.23.0` pins from Step 2. |
| `AttributeError: module 'cv2' has no attribute 'aruco'` | `opencv-python` got installed alongside `opencv-contrib-python`. Run `pip uninstall -y opencv-python` then `pip install --force-reinstall 'opencv-contrib-python==4.10.*'`. |
| `ImportError: numpy.core.multiarray failed to import` | `numpy` was upgraded past 1.26. Run `pip install --force-reinstall 'numpy==1.26.*'`. |
| Ultralytics auto-install fails during export: `No matching distribution found for onnxruntime-gpu` | `onnxruntime-gpu` has no `aarch64` wheel on PyPI. Install manually per Step 3: `pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnx onnxslim onnxruntime-gpu` |
| Engine export hangs or OOMs | Close other GPU/CPU workloads. The builder can use >3 GB on Orin Nano. Run `sudo jetson_clocks` to pin max clocks. |
| `NvMapMemAllocInternalTagged: ... error 12` spam during TRT export | **Not fatal.** NVMAP can't allocate the requested buffer size; TRT retries smaller and continues. Usually caused by a concurrent build or other memory-heavy workload. |
| `[TRT] [E] Error Code: 9: Skipping tactic ... Cask Gemm` repeated during engine build | **Not fatal.** TRT is skipping kernel variants that can't allocate workspace under memory pressure. Engine still builds with the surviving tactics. Rebuild on a quiet system for the optimal engine. |
| `Could not load library libnvinfer.so.X` when loading an engine | JetPack version mismatch. The `.engine` was built with a different TRT version. Re-export on *this* Jetson. |

---

## Quick reference — the whole flow, no explanation

```bash
# Step 0: verify your system
cat /etc/nv_tegra_release           # → R36, REVISION: 4.7
/usr/local/cuda/bin/nvcc --version  # → release 12.6
python3 --version                   # → Python 3.10.12

# Step 1: baseline pins
cd <your-checkout-of-this-repo>
pip install -r Project5_WS/requirements.txt

# Step 2: Jetson-native torch
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.8.0 torchvision==0.23.0

# Step 3: ultralytics (no-deps) + manual deps + export-time deps
pip install --no-deps ultralytics
pip install matplotlib pillow tqdm psutil py-cpuinfo pandas seaborn \
    requests ultralytics-thop
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    onnx onnxslim onnxruntime-gpu

# Safety net: re-pin numpy + opencv-contrib in case anything drifted
pip install --force-reinstall 'numpy==1.26.*' 'opencv-contrib-python==4.10.*'
pip uninstall -y opencv-python 2>/dev/null

# Verify
python3 -c "
import numpy, cv2, torch
from ultralytics import YOLO
print('numpy', numpy.__version__, '| cv2', cv2.__version__,
      '| aruco', 'aruco' in dir(cv2), '| cuda', torch.cuda.is_available())
"

# Step 5: export TensorRT FP16 engine (3–10 min on Orin Nano)
sudo jetson_clocks   # optional: pin max clocks for faster + cleaner build
python3 Project6_Class_Distro/src/vision_processing_package/models/pt_to_engine.py
```
