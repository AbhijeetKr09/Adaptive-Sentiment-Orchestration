"""
patch_notebook.py
=================
Patches ASO_Pipeline_SPIT.ipynb with the production-grade SPIT v2 logic:

  Cell 1  — adds psutil to pip install
  Cell 2  — enriched device banner (CPU/GPU info + adaptive batch size)
  Cell 7c — replaces SPIT module with §1-§7 production implementation:
              * 6-element hardware health vector  (cpu_load, ram, cpu_temp,
                swap, gpu_temp, vram)
              * Real psutil CPU telemetry (no simulation fallback)
              * CPU-aware PID (pseudo-thermal mapping from CPU load %)
              * RoutingDecision carries hardware_snapshot
  Cell 7g — updates SPIT inference loop to use new 6-element API
             and new SPITConfig fields (T_cpu_setpoint, RAM_capacity_gb)
  Title   — updates GPU-only note to CPU/GPU
"""

import json

NB   = "ASO_Pipeline_SPIT.ipynb"

with open(NB, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

cells = nb["cells"]

def get(cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  TITLE CELL
# ──────────────────────────────────────────────────────────────────────────────
get("title-cell")["source"] = [
    "# Adaptive Sentiment Orchestration (ASO) + SPIT\n",
    "### Hybrid Framework with Dynamic Socio-Physical Thresholding\n",
    "\n",
    "---\n",
    "**Paper:** *Adaptive Sentiment Orchestration via Socio-Physical Integrated "
    "Thresholding (SPIT) for High-Velocity Social Streams*\n",
    "\n",
    "**Architecture:**\n",
    "```\n",
    "Input Text\n",
    "    |\n",
    "    v\n",
    "[ Tier-1: DistilBERT ] --> confidence c + token_ids\n",
    "    |\n",
    "    v\n",
    "[ SPIT: Dynamic Threshold (NOVEL) ]\n",
    "  Hardware Telemetry Sidecar (GPU *or* CPU — real sensors, no simulation)\n",
    "    cpu_load, ram, cpu_temp, swap  ←  psutil  (always)\n",
    "    gpu_temp, vram                 ←  pynvml  (when GPU present)\n",
    "  PID --> Phi_physical\n",
    "  Socio Signals --> Psi_socio\n",
    "  Token IDs     --> H_token\n",
    "  tau = sigmoid(Phi*(1-Psi)+H)*(tau_max-tau_min)+tau_min\n",
    "    |\n",
    "  conf >= tau_dynamic?\n",
    "  YES --> DistilBERT (Tier-1)  |  NO --> BERT (Tier-2)\n",
    "```\n",
    "\n",
    "**5 Models evaluated:**\n",
    "1. Logistic Regression (TF-IDF baseline)\n",
    "2. DistilBERT (Tier-1 only)\n",
    "3. BERT (Tier-2 only)\n",
    "4. ASO Hybrid — fixed tau=0.85\n",
    "5. **ASO+SPIT** — dynamic tau (this paper)\n",
    "\n",
    "---\n",
    "> **Device:** Runs on **GPU** (T4 recommended) **or CPU**. "
    "Pipeline auto-detects hardware and adapts batch size and telemetry accordingly.  \n",
    "> Colab: Runtime → Change runtime type → T4 GPU *(optional, CPU is fully supported)*",
]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  INSTALL CELL  (add psutil)
# ──────────────────────────────────────────────────────────────────────────────
get("cell-install")["source"] = [
    "# ============================================================\n",
    "# CELL 1: Install Dependencies  (run once per Colab session)\n",
    "# ============================================================\n",
    "!pip install -q transformers datasets scikit-learn torch pandas matplotlib\n",
    "!pip install -q accelerate sentencepiece pynvml psutil\n",
    'print("All dependencies installed.")',
]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  CONFIG CELL  (richer device banner + adaptive batch size)
# ──────────────────────────────────────────────────────────────────────────────
get("cell-config")["source"] = [
    "# ============================================================\n",
    "# CELL 2: Global Configuration & Imports\n",
    "# ============================================================\n",
    "import os, sys, logging, warnings, platform\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')\n",
    "\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "\n",
    "# ── Device Detection ────────────────────────────────────────\n",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "ON_GPU = DEVICE == 'cuda'\n",
    "\n",
    "print('=' * 58)\n",
    "print('  HARDWARE ENVIRONMENT')\n",
    "print('=' * 58)\n",
    "print(f'  PyTorch version : {torch.__version__}')\n",
    "print(f'  Device          : {DEVICE.upper()}')\n",
    "\n",
    "if ON_GPU:\n",
    "    _gp = torch.cuda.get_device_properties(0)\n",
    "    print(f'  GPU Name        : {_gp.name}')\n",
    "    print(f'  GPU VRAM        : {_gp.total_memory/1024**3:.1f} GB')\n",
    "    print(f'  CUDA version    : {torch.version.cuda}')\n",
    "else:\n",
    "    try:\n",
    "        import psutil as _ps\n",
    "        _cpu_phys    = _ps.cpu_count(logical=False)\n",
    "        _cpu_logical = _ps.cpu_count(logical=True)\n",
    "        _ram_gb      = _ps.virtual_memory().total / 1024**3\n",
    "        print(f'  CPU             : {platform.processor() or platform.machine()}')\n",
    "        print(f'  CPU Cores       : {_cpu_phys} physical / {_cpu_logical} logical')\n",
    "        print(f'  System RAM      : {_ram_gb:.1f} GB')\n",
    "    except ImportError:\n",
    "        print('  [psutil not installed yet — run Cell 1 first]')\n",
    "    print('  NOTE: Running on CPU — inference is slower but fully functional.')\n",
    "    print('  NOTE: SPIT uses real CPU telemetry (psutil) — no simulation.')\n",
    "print('=' * 58)\n",
    "\n",
    "# Dataset\n",
    "DATA_SOURCE   = 'sst2'\n",
    "MAX_SAMPLES   = 4000\n",
    "TEST_SIZE     = 0.20\n",
    "CSV_PATH      = None\n",
    "\n",
    "# Models\n",
    "TIER1_MODEL   = 'distilbert-base-uncased-finetuned-sst-2-english'\n",
    "TIER2_MODEL   = 'textattack/bert-base-uncased-SST-2'\n",
    "\n",
    "# ASO (fixed threshold)\n",
    "ASO_THRESHOLD = 0.85\n",
    "\n",
    "# SPIT configuration\n",
    "SPIT_T_SETPOINT     = 85.0   # GPU thermal set-point °C\n",
    "SPIT_T_CPU_SETPOINT = 90.0   # CPU thermal set-point °C\n",
    "SPIT_VRAM_GB        = 16.0   # Colab T4 VRAM\n",
    "SPIT_KP             = 0.008\n",
    "SPIT_KI             = 0.002\n",
    "SPIT_KD             = 0.004\n",
    "SPIT_TAU_MIN        = 0.50\n",
    "SPIT_TAU_MAX        = 0.92\n",
    "\n",
    "# Auto-detect system RAM for CPU mode\n",
    "try:\n",
    "    import psutil as _ps2\n",
    "    SPIT_RAM_GB = _ps2.virtual_memory().total / 1024**3\n",
    "except Exception:\n",
    "    SPIT_RAM_GB = 12.0   # Colab default\n",
    "\n",
    "# Adaptive inference batch size\n",
    "INFERENCE_BATCH_SIZE = 32 if ON_GPU else 8\n",
    "\n",
    "print(f'\\n  Tier-1 model       : {TIER1_MODEL}')\n",
    "print(f'  Tier-2 model       : {TIER2_MODEL}')\n",
    "print(f'  ASO tau (fixed)    : {ASO_THRESHOLD}')\n",
    "print(f'  SPIT tau range     : [{SPIT_TAU_MIN}, {SPIT_TAU_MAX}]')\n",
    "print(f'  SPIT T_setpoint    : {SPIT_T_SETPOINT} °C (GPU) / {SPIT_T_CPU_SETPOINT} °C (CPU)')\n",
    "print(f'  RAM detected       : {SPIT_RAM_GB:.1f} GB')\n",
    "print(f'  Inference batch    : {INFERENCE_BATCH_SIZE} samples/batch')",
]

# ──────────────────────────────────────────────────────────────────────────────
# 4.  SPIT MODULE CELL  (full §1-§7 production implementation)
# ──────────────────────────────────────────────────────────────────────────────
SPIT_MODULE_SOURCE = '''\
# ============================================================
# CELL 7c: SPIT Module — §1-§7 Production Implementation
# GPU + CPU aware | real hardware telemetry | no simulation
# ============================================================
import numpy as np
import threading
import time as _time
import warnings as _warnings
from dataclasses import dataclass as _dc, field as _field
from typing import Optional as _Opt
from enum import Enum as _Enum

# ── psutil (CPU telemetry — always first) ────────────────────
try:
    import psutil as _psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    _warnings.warn(
        "[SPIT] psutil not available. CPU telemetry will be zeroed. "
        "Install with: !pip install psutil",
        RuntimeWarning
    )

# ── pynvml (GPU telemetry — only when GPU present) ───────────
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ── Announce telemetry mode ───────────────────────────────────
if NVML_AVAILABLE:
    _TELEMETRY_MODE = "GPU+CPU"
elif PSUTIL_AVAILABLE:
    _TELEMETRY_MODE = "CPU"
else:
    _TELEMETRY_MODE = "UNAVAILABLE"

print(f"[SPIT] Telemetry mode: {_TELEMETRY_MODE}")


# =============================================================
# §1  Enumerations & Config
# =============================================================

class PlatformType(_Enum):
    """Platform-level socio-entropy H_socio. Higher → more ambiguity → lower tau."""
    TWITTER   = 0.35
    REDDIT    = 0.45
    INSTAGRAM = 0.60
    TIKTOK    = 0.75
    UNKNOWN   = 0.50

@_dc
class SPITConfig:
    # Physical set-points
    T_setpoint:       float = 85.0     # GPU thermal set-point (°C)
    T_cpu_setpoint:   float = 90.0     # CPU thermal set-point (°C)
    T_min:            float = 40.0     # Idle temperature floor (°C)
    VRAM_capacity:    float = 16.0     # Total VRAM (GB)
    RAM_capacity_gb:  float = 12.0     # Total system RAM (GB)
    # PID gains
    Kp: float = 0.008
    Ki: float = 0.002
    Kd: float = 0.004
    # Threshold bounds
    tau_min: float = 0.50
    tau_max: float = 0.92
    # Socio caps
    retweet_cap:      float = 1_000_000.0
    follower_cap:     float = 10_000_000.0
    burst_cap:        float = 500.0
    # Sidecar
    poll_interval_ms: float = 20.0
    integral_limit:   float = 50.0


# =============================================================
# §2  Hardware Telemetry Sidecar
# =============================================================

class HardwareTelemetrySidecar:
    """
    Background thread polling real hardware at poll_interval_ms cadence.
    Exposes a lock-free 6-element health vector:

      h[0] cpu_load_norm    (psutil, always)
      h[1] ram_norm         (psutil, always)
      h[2] cpu_temp_norm    (psutil sensors — 0.0 if VM doesn't expose them)
      h[3] swap_norm        (psutil, always)
      h[4] gpu_temp_norm    (pynvml  — 0.0 if no GPU)
      h[5] vram_norm        (pynvml  — 0.0 if no GPU)

    No simulation. Values are real hardware readings.
    """

    VECTOR_LABELS = [
        "cpu_load_norm", "ram_usage_norm", "cpu_temp_norm",
        "swap_norm", "gpu_temp_norm", "vram_usage_norm",
    ]

    def __init__(self, config: SPITConfig):
        self.config = config
        self._lock  = threading.Lock()
        self._health_vector = np.zeros(6, dtype=np.float32)
        self._running = False
        self._thread: _Opt[threading.Thread] = None
        self._gpu_handle = None
        if NVML_AVAILABLE:
            try:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass
        self._total_ram_gb = (
            _psutil.virtual_memory().total / (1024 ** 3)
            if PSUTIL_AVAILABLE else config.RAM_capacity_gb
        )

    def start(self):
        h = self._read_hardware()
        with self._lock:
            self._health_vector = h
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="SPIT-Telemetry"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._health_vector.copy()

    def read_labeled(self) -> dict:
        h = self.read()
        return {lbl: float(h[i]) for i, lbl in enumerate(self.VECTOR_LABELS)}

    def _poll_loop(self):
        iv = self.config.poll_interval_ms / 1000.0
        while self._running:
            h = self._read_hardware()
            with self._lock:
                self._health_vector = h
            _time.sleep(iv)

    def _read_hardware(self) -> np.ndarray:
        return np.concatenate([self._read_cpu(), self._read_gpu()]).astype(np.float32)

    def _read_cpu(self) -> np.ndarray:
        if not PSUTIL_AVAILABLE:
            return np.zeros(4, dtype=np.float32)
        try:
            cpu_load_norm = np.clip(_psutil.cpu_percent(interval=None) / 100.0, 0.0, 1.0)
            vm  = _psutil.virtual_memory()
            ram_norm = np.clip(vm.used / vm.total, 0.0, 1.0)
            cpu_temp_norm = self._read_cpu_temp()
            swap = _psutil.swap_memory()
            swap_norm = np.clip(swap.used / swap.total, 0.0, 1.0) if swap.total > 0 else 0.0
            return np.array([cpu_load_norm, ram_norm, cpu_temp_norm, swap_norm], dtype=np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _read_cpu_temp(self) -> float:
        try:
            sensors = _psutil.sensors_temperatures()
            if not sensors:
                return 0.0
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz", "cpu-thermal"):
                if key in sensors:
                    temps = [r.current for r in sensors[key] if r.current is not None]
                    if temps:
                        return float(np.clip(max(temps) / self.config.T_cpu_setpoint, 0.0, 1.0))
            for entries in sensors.values():
                temps = [r.current for r in entries if r.current is not None]
                if temps:
                    return float(np.clip(max(temps) / self.config.T_cpu_setpoint, 0.0, 1.0))
        except Exception:
            pass
        return 0.0

    def _read_gpu(self) -> np.ndarray:
        if not (NVML_AVAILABLE and self._gpu_handle is not None):
            return np.zeros(2, dtype=np.float32)
        try:
            temp_c    = _pynvml.nvmlDeviceGetTemperature(self._gpu_handle, _pynvml.NVML_TEMPERATURE_GPU)
            mem_info  = _pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            vram_gb   = mem_info.used / (1024 ** 3)
            return np.array([
                np.clip(temp_c / self.config.T_setpoint, 0.0, 1.0),
                np.clip(vram_gb / self.config.VRAM_capacity, 0.0, 1.0),
            ], dtype=np.float32)
        except Exception:
            return np.zeros(2, dtype=np.float32)


# =============================================================
# §3  PID Controller  (CPU-aware)
# =============================================================

class PIDController:
    """
    Maps the 6-element health vector → Φ_physical ∈ [0, 1].

    GPU mode:  primary error = GPU temperature vs T_setpoint (°C)
    CPU mode:  primary error = CPU load re-expressed as pseudo-temperature
               so that the same PID gains remain numerically valid.
               CPU temperature is added as secondary when exposed by the VM.

    Φ_physical → 0  hardware relaxed (threshold can be lower / more diligent)
    Φ_physical → 1  hardware stressed (threshold raised / more frugal)
    """
    def __init__(self, config: SPITConfig):
        self.Kp = config.Kp; self.Ki = config.Ki; self.Kd = config.Kd
        self.T_setpoint     = config.T_setpoint
        self.T_cpu_setpoint = config.T_cpu_setpoint
        self.integral_limit = config.integral_limit
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_time:  float = _time.time()

    def step(self, health_vector: np.ndarray) -> float:
        """
        health_vector shape (6,):
            [cpu_load, ram, cpu_temp, swap, gpu_temp, vram]
        """
        now = _time.time()
        dt  = max(now - self._prev_time, 1e-6)

        cpu_load_norm = float(health_vector[0])
        ram_norm      = float(health_vector[1])
        cpu_temp_norm = float(health_vector[2])
        swap_norm     = float(health_vector[3])
        gpu_temp_norm = float(health_vector[4])
        vram_norm     = float(health_vector[5])

        gpu_present = gpu_temp_norm > 0.0

        if gpu_present:
            T_gpu    = gpu_temp_norm * self.T_setpoint
            error    = T_gpu - self.T_setpoint
            secondary = 0.08 * vram_norm + 0.04 * ram_norm
        else:
            # CPU pseudo-thermal mapping: 100% load ≡ T_cpu_setpoint
            T_cpu_equiv = cpu_load_norm * self.T_cpu_setpoint
            error       = T_cpu_equiv - self.T_cpu_setpoint
            cpu_temp_contrib = cpu_temp_norm * self.T_cpu_setpoint if cpu_temp_norm > 0 else 0.0
            secondary = (
                0.08 * ram_norm
                + 0.05 * swap_norm
                + 0.04 * (cpu_temp_contrib / self.T_cpu_setpoint)
            )

        P = self.Kp * error
        self._integral = np.clip(
            self._integral + error * dt, -self.integral_limit, self.integral_limit
        )
        I = self.Ki * self._integral
        D = self.Kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = now
        return float(np.clip(P + I + D + secondary, 0.0, 1.0))


# =============================================================
# §4  Ψ_socio
# =============================================================

def compute_psi_socio(rt, fc, bpm, plat, cfg):
    V = np.clip(rt / cfg.retweet_cap, 0.0, 1.0)
    A = (np.clip(np.log1p(fc) / np.log1p(cfg.follower_cap), 0.0, 1.0) if fc > 0 else 0.0)
    B = np.clip(bpm / cfg.burst_cap, 0.0, 1.0)
    H = plat.value
    return float(np.clip(np.dot([0.30, 0.30, 0.25, 0.15], [V, A, B, H]), 0.0, 1.0))


# =============================================================
# §5  H_token
# =============================================================

def compute_h_token(ids, vsz=30522):
    if not ids:
        return 0.0
    _, c = np.unique(np.array(ids, dtype=np.int32), return_counts=True)
    p = c / c.sum()
    return float(np.clip(-np.sum(p * np.log(p + 1e-12)) / np.log(vsz + 1e-12), 0.0, 1.0))


# =============================================================
# §6  SPIT Function
# =============================================================

def spit(phi_physical, retweet_count, follower_count, burst_rate_ppm,
         platform, token_ids, config=None):
    """
    τ_dynamic = σ( Φ_physical·(1−Ψ_socio) + H_token ) · (τ_max−τ_min) + τ_min

    Returns dict of all components (for logging/ablation).
    """
    if config is None:
        config = SPITConfig()
    psi   = compute_psi_socio(retweet_count, follower_count, burst_rate_ppm, platform, config)
    h     = compute_h_token(token_ids)
    pt    = phi_physical * (1.0 - psi)
    z     = pt + h
    sg    = 1.0 / (1.0 + np.exp(-z))
    tau   = float(np.clip(sg * (config.tau_max - config.tau_min) + config.tau_min,
                          config.tau_min, config.tau_max))
    return {
        "tau_dynamic":   tau,
        "phi_physical":  phi_physical,
        "psi_socio":     psi,
        "h_token":       h,
        "physical_term": pt,
        "sigmoid_input": z,
    }


# =============================================================
# §7  Data Classes & Cascade Router
# =============================================================

@_dc
class PostContext:
    text:            str
    token_ids:       list
    distilbert_conf: float
    retweet_count:   float = 0.0
    follower_count:  float = 0.0
    burst_rate_ppm:  float = 0.0
    platform:        PlatformType = PlatformType.UNKNOWN

@_dc
class RoutingDecision:
    escalate:          bool
    tau_dynamic:       float
    distilbert_conf:   float
    spit_components:   dict
    hardware_snapshot: dict           # labeled telemetry at decision time
    tier: str = _field(init=False)
    def __post_init__(self):
        self.tier = "BERT (Tier-2)" if self.escalate else "DistilBERT (Tier-1)"

class SPITCascadeRouter:
    def __init__(self, config=None):
        if config is None:
            config = SPITConfig()
        self.config  = config
        self.sidecar = HardwareTelemetrySidecar(config)
        self.pid     = PIDController(config)
        self._started = False

    def start(self):
        self.sidecar.start()
        self._started = True
        print("[SPIT] Cascade router started. Telemetry sidecar active.")

    def stop(self):
        self.sidecar.stop()
        print("[SPIT] Cascade router stopped.")

    def route(self, post: PostContext) -> RoutingDecision:
        if not self._started:
            raise RuntimeError("Call router.start() before routing posts.")
        h   = self.sidecar.read()          # shape (6,)
        phi = self.pid.step(h)
        r   = spit(phi, post.retweet_count, post.follower_count,
                   post.burst_rate_ppm, post.platform, post.token_ids, self.config)
        return RoutingDecision(
            escalate          = post.distilbert_conf < r["tau_dynamic"],
            tau_dynamic       = r["tau_dynamic"],
            distilbert_conf   = post.distilbert_conf,
            spit_components   = r,
            hardware_snapshot = self.sidecar.read_labeled(),
        )


print(f"[SPIT] Module loaded | Telemetry: {_TELEMETRY_MODE} | Device: {DEVICE.upper()}")
'''

get("cell-spit-module")["source"] = [
    line + ("\n" if not line.endswith("\n") else "")
    for line in SPIT_MODULE_SOURCE.splitlines(keepends=True)
]

# ──────────────────────────────────────────────────────────────────────────────
# 5.  SPIT INFERENCE CELL  (update to 6-element health vector API)
# ──────────────────────────────────────────────────────────────────────────────
SPIT_INF_SOURCE = '''\
# ============================================================
# CELL 7g: SPIT Inference on Full Test Set
# ============================================================
import time

# -- Build SPITConfig from global settings --
spit_config = SPITConfig(
    T_setpoint      = SPIT_T_SETPOINT,
    T_cpu_setpoint  = SPIT_T_CPU_SETPOINT,
    VRAM_capacity   = SPIT_VRAM_GB,
    RAM_capacity_gb = SPIT_RAM_GB,
    Kp=SPIT_KP, Ki=SPIT_KI, Kd=SPIT_KD,
    tau_min=SPIT_TAU_MIN, tau_max=SPIT_TAU_MAX,
)
print(f"[SPIT] Config: device={DEVICE.upper()}, "
      f"T_setpoint={spit_config.T_setpoint}°C (GPU) / "
      f"{spit_config.T_cpu_setpoint}°C (CPU), "
      f"resource_cap={'VRAM '+str(SPIT_VRAM_GB)+'GB' if ON_GPU else 'RAM '+str(round(SPIT_RAM_GB,1))+'GB'}")

# Step 1: Tokenise for H_token
print("[SPIT] Tokenising test set ...")
all_token_ids = [
    t1_tokenizer.encode(txt, truncation=True, max_length=128)
    for txt in X_test
]

# Step 2: Start router
spit_router = SPITCascadeRouter(spit_config)
spit_router.start()
time.sleep(0.1)   # let sidecar pre-read hardware

# Step 3: Compute tau_dynamic per sample via sidecar + PID
print("[SPIT] Computing dynamic thresholds ...")
spit_tau_vals=[]; spit_phi_vals=[]; spit_psi_vals=[]; spit_htoken_vals=[]
spit_hw_snapshots = []    # hardware telemetry at decision time (for ablation)

for start in range(0, len(X_test), INFERENCE_BATCH_SIZE):
    end   = min(start + INFERENCE_BATCH_SIZE, len(X_test))
    h_vec = spit_router.sidecar.read()   # 6-element health vector
    phi   = spit_router.pid.step(h_vec)  # CPU-aware PID

    for i in range(start, end):
        r = spit(phi, float(synth_retweets[i]), float(synth_followers[i]),
                 float(synth_burst[i]), synth_platform_enum,
                 all_token_ids[i], spit_config)
        spit_tau_vals.append(r["tau_dynamic"])
        spit_phi_vals.append(r["phi_physical"])
        spit_psi_vals.append(r["psi_socio"])
        spit_htoken_vals.append(r["h_token"])

# Capture one hardware snapshot for reporting
hw_snap = spit_router.sidecar.read_labeled()
spit_router.stop()

# Step 4: Escalation mask
esc_mask = [c < t for c, t in zip(t1_confs, spit_tau_vals)]
esc_idx  = [i for i, m in enumerate(esc_mask) if m]
esc_txts = [X_test[i] for i in esc_idx]
print(f"[SPIT] Escalating {len(esc_idx)}/{len(X_test)} ({len(esc_idx)/len(X_test):.1%}) to Tier-2 ...")

# Step 5: Batch Tier-2 for escalated samples
t2map={}; t2lat={}
if esc_txts:
    for s in range(0, len(esc_txts), INFERENCE_BATCH_SIZE):
        bt=esc_txts[s:s+INFERENCE_BATCH_SIZE]; bi=esc_idx[s:s+INFERENCE_BATCH_SIZE]
        t0=time.perf_counter(); pb,_=t2_predict_batch(bt); lat=(time.perf_counter()-t0)/len(bt)
        for idx, p in zip(bi, pb): t2map[idx]=p; t2lat[idx]=lat

# Step 6: Assemble predictions + latencies
spit_preds=[]; spit_latencies=[]; spit_tier2_count=len(esc_idx)
for i in range(len(X_test)):
    if i in t2map: spit_preds.append(t2map[i]); spit_latencies.append(t1_latencies[i]+t2lat[i])
    else:          spit_preds.append(t1_preds[i]); spit_latencies.append(t1_latencies[i])

# Step 7: Metrics
from sklearn.metrics import accuracy_score, f1_score
spit_acc = accuracy_score(y_test, spit_preds)
spit_f1  = f1_score(y_test, spit_preds, average="macro", zero_division=0)
spit_lat = float(np.mean(spit_latencies)) * 1000
spit_tier2_rate = spit_tier2_count / len(X_test)

print(f"\\n{'='*58}")
print(f"  ASO+SPIT Results  |  Telemetry: {_TELEMETRY_MODE}")
print(f"{'='*58}")
print(f"  Accuracy     : {spit_acc:.4f}")
print(f"  F1 (Macro)   : {spit_f1:.4f}")
print(f"  Avg latency  : {spit_lat:.4f} ms/sample")
print(f"  Tier-2 rate  : {spit_tier2_count}/{len(X_test)} ({spit_tier2_rate:.1%})")
print(f"  tau range    : [{min(spit_tau_vals):.4f}, {max(spit_tau_vals):.4f}]")
print(f"  tau mean±std : {np.mean(spit_tau_vals):.4f} ± {np.std(spit_tau_vals):.4f}")
print(f"{'='*58}")
print(f"  Hardware telemetry snapshot (last batch):")
for k, v in hw_snap.items():
    note = ""
    if k == "cpu_temp_norm" and v == 0.0:
        note = "  ← unavailable on this VM (normal)"
    elif k in ("gpu_temp_norm", "vram_usage_norm") and v == 0.0 and not ON_GPU:
        note = "  ← no GPU"
    print(f"    {k:<22}: {v*100:5.1f} %{note}")
print(f"{'='*58}")
'''

get("cell-spit-inference")["source"] = [
    line + ("\n" if not line.endswith("\n") else "")
    for line in SPIT_INF_SOURCE.splitlines(keepends=True)
]

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f"\n✓  {NB} patched successfully.")
print("   Changes applied:")
print("   [title]   Updated GPU-only note → CPU/GPU supported")
print("   [Cell 1]  Added psutil to pip install")
print("   [Cell 2]  Rich device banner: CPU cores/RAM + adaptive batch size")
print("   [Cell 7c] SPIT module v2: 6-element health vector, CPU-aware PID,")
print("             real psutil telemetry — no simulation fallback")
print("   [Cell 7g] Inference loop: sidecar.read() → 6-elem vector,")
print("             pid.step(h_vec), SPITConfig with T_cpu_setpoint+RAM_capacity_gb,")
print("             hardware_snapshot printed in results table")
