"""
spit.py
=======
Socio-Physical Integrated Thresholding (SPIT)
Adaptive Sentiment Orchestration for High-Velocity Social Media Streams

Architecture:  Control-Theoretic PID Feedback Loop
               + Vectorised Rule Engine
               + Asynchronous Telemetry Sidecar

Hardware awareness:
    GPU mode  — pynvml : real GPU temperature (°C) + VRAM usage
    CPU mode  — psutil : real CPU temperature (°C) or CPU% proxy + RAM usage
    Simulated — sine-wave temperature + sawtooth memory (no libraries needed)

Authors:       ASO Research Team, IILM University
Year:          2026
Paper:         "Adaptive Sentiment Orchestration via Socio-Physical
                Integrated Thresholding for High-Velocity Social Streams"
"""

import numpy as np
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ── Optional GPU telemetry ────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    warnings.warn(
        "[SPIT] pynvml not available. GPU telemetry disabled."
        " Install with: !pip install pynvml",
        RuntimeWarning,
    )

# ── Optional CPU telemetry ────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False
    warnings.warn(
        "[SPIT] psutil not available. CPU telemetry disabled."
        " Install with: !pip install psutil",
        RuntimeWarning,
    )


# =============================================================================
# §0  COMPUTE MODE DETECTION
# =============================================================================

class ComputeMode(Enum):
    """
    Auto-detected execution context, determining which telemetry backend
    the HardwareTelemetrySidecar will use.

    GPU   — pynvml available + CUDA device found  → real GPU temp + VRAM
    CPU   — psutil available, no CUDA             → real CPU temp + RAM
    SIM   — neither library available             → realistic simulation
    """
    GPU = "gpu"
    CPU = "cpu"
    SIM = "sim"


def detect_compute_mode() -> ComputeMode:
    """
    Probe the runtime environment and return the appropriate ComputeMode.

    Detection order:
        1. If pynvml is available AND at least one CUDA device is present → GPU
        2. If psutil is available                                          → CPU
        3. Otherwise                                                       → SIM

    Returns:
        ComputeMode enum member
    """
    if NVML_AVAILABLE:
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                return ComputeMode.GPU
        except Exception:
            pass
    if PSUTIL_AVAILABLE:
        return ComputeMode.CPU
    return ComputeMode.SIM


# =============================================================================
# §1  ENUMERATIONS & CONFIGURATION
# =============================================================================

class PlatformType(Enum):
    """
    Platform-level socio-entropy priors (H_socio).

    Higher entropy → higher baseline ambiguity → lower escalation threshold.
    Multimodal platforms (TikTok, Reels) carry maximum sarcasm/irony risk.
    """
    TWITTER   = 0.35   # Short-form, high velocity, moderate irony
    REDDIT    = 0.45   # Thread-based, niche slang, moderate-high entropy
    INSTAGRAM = 0.60   # Visual-text misalignment, sarcasm-heavy captions
    TIKTOK    = 0.75   # Multimodal frontier: audio-text sentiment divergence
    UNKNOWN   = 0.50   # Conservative fallback


@dataclass
class SPITConfig:
    """
    All tunable hyperparameters for the SPIT function.
    Centralised here so ablation studies only touch this dataclass.
    """
    # ── GPU physical safety set-points ─────────────────────────────────────
    T_setpoint:    float = 85.0    # GPU temperature safety set-point (°C)
    T_min:         float = 40.0    # Idle GPU temperature floor (°C)
    VRAM_capacity: float = 16.0    # Total VRAM in GB (Colab T4 = 16 GB)

    # ── CPU physical safety set-points (used when no GPU is present) ───────
    cpu_temp_setpoint: float = 75.0   # CPU temperature safety set-point (°C)
    ram_capacity_gb:   float = 12.8   # Colab standard RAM (GB); adjust if needed

    # ── PID controller gains ───────────────────────────────────────────────
    Kp: float = 0.008   # Proportional: react to current thermal error
    Ki: float = 0.002   # Integral:     penalise sustained load accumulation
    Kd: float = 0.004   # Derivative:   anticipate slope of incoming burst

    # ── Threshold bounds (τ always clipped to this range) ─────────────────
    tau_min: float = 0.50   # Floor: never escalate everything blindly
    tau_max: float = 0.92   # Ceiling: never block escalation entirely

    # ── Socio-variable normalisation caps ──────────────────────────────────
    retweet_cap:    float = 1_000_000.0   # Normalise virality to [0,1]
    follower_cap:   float = 10_000_000.0  # Normalise authority to [0,1]
    burst_cap:      float = 500.0         # Posts/minute considered full burst

    # ── Telemetry sidecar ──────────────────────────────────────────────────
    poll_interval_ms: float = 20.0   # Hardware poll cadence (milliseconds)

    # ── PID integral anti-windup ───────────────────────────────────────────
    integral_limit: float = 50.0


# =============================================================================
# §2  HARDWARE TELEMETRY SIDECAR
# =============================================================================

class HardwareTelemetrySidecar:
    """
    Asynchronous background thread that polls hardware sensors at a fixed
    cadence and exposes a lock-free 'health vector' to the inference thread.

    Pattern:  Asynchronous Telemetry Sidecar
    Latency:  ~20 ms poll cadence; main thread reads in O(1).

    Auto-detects compute mode at initialisation:

        GPU mode (pynvml + CUDA device available)
            h[0] = GPU temperature  / T_setpoint        ∈ [0, 1]
            h[1] = VRAM used (GB)   / VRAM_capacity     ∈ [0, 1]

        CPU mode (psutil available, no GPU)
            h[0] = CPU temperature  / cpu_temp_setpoint  ∈ [0, 1]
                   (falls back to normalised CPU% if sensors unavailable)
            h[1] = RAM used (GB)    / ram_capacity_gb    ∈ [0, 1]

        SIM mode (no telemetry libraries)
            h[0] = sinusoidal temperature simulation     ∈ [0, 1]
            h[1] = sawtooth memory simulation            ∈ [0, 1]
    """

    def __init__(self, config: SPITConfig):
        self.config = config
        self._lock  = threading.Lock()
        self._health_vector = np.array([0.3, 0.3], dtype=np.float32)
        self._sim_time = 0.0
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        # ── Detect compute mode ────────────────────────────────────────────
        self.mode: ComputeMode = detect_compute_mode()
        self._gpu_handle = None

        if self.mode == ComputeMode.GPU:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                # GPU detection failed at handle level — fall back to CPU/SIM
                self.mode = ComputeMode.CPU if PSUTIL_AVAILABLE else ComputeMode.SIM

        # Resolve effective set-point for PID (mode-dependent)
        self.effective_setpoint = (
            config.T_setpoint if self.mode == ComputeMode.GPU
            else config.cpu_temp_setpoint
        )

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        """Launch the sidecar thread (non-blocking)."""
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="SPIT-Telemetry"
        )
        self._thread.start()
        mode_str = {
            ComputeMode.GPU: f"GPU (pynvml) — setpoint {self.config.T_setpoint}°C",
            ComputeMode.CPU: f"CPU (psutil) — setpoint {self.config.cpu_temp_setpoint}°C",
            ComputeMode.SIM: "Simulated (no telemetry library)",
        }[self.mode]
        print(f"[SPIT] Telemetry sidecar started | mode: {mode_str}")

    def stop(self):
        """Gracefully shut down the sidecar."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def read(self) -> np.ndarray:
        """
        Return a copy of the current health vector.
        Shape: (2,) → [T_norm, Memory_norm]
        Lock-protected; O(1) for the caller.
        """
        with self._lock:
            return self._health_vector.copy()

    # ── Internal poll loop ────────────────────────────────────────────────

    def _poll_loop(self):
        interval = self.config.poll_interval_ms / 1000.0
        while self._running:
            h = self._read_hardware()
            with self._lock:
                self._health_vector = h
            time.sleep(interval)

    def _read_hardware(self) -> np.ndarray:
        """Dispatch to the correct backend based on detected compute mode."""
        if self.mode == ComputeMode.GPU:
            return self._read_gpu()
        elif self.mode == ComputeMode.CPU:
            return self._read_cpu()
        else:
            return self._read_simulated()

    # ── GPU backend (pynvml) ──────────────────────────────────────────────

    def _read_gpu(self) -> np.ndarray:
        """Read real GPU temperature and VRAM from pynvml."""
        try:
            temp_c = pynvml.nvmlDeviceGetTemperature(
                self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            mem_info     = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            vram_used_gb = mem_info.used / (1024 ** 3)
            T_norm    = np.clip(temp_c / self.config.T_setpoint, 0.0, 1.0)
            MEM_norm  = np.clip(vram_used_gb / self.config.VRAM_capacity, 0.0, 1.0)
            return np.array([T_norm, MEM_norm], dtype=np.float32)
        except Exception:
            return np.array([0.5, 0.5], dtype=np.float32)

    # ── CPU backend (psutil) ──────────────────────────────────────────────

    def _read_cpu(self) -> np.ndarray:
        """
        Read CPU temperature and RAM usage via psutil.

        Temperature strategy (in priority order):
            1. psutil.sensors_temperatures() — coretemp (Intel) / k10temp (AMD)
               / cpu_thermal (ARM/Colab TPU) / acpitz (ACPI zone)
            2. CPU utilisation % as a load-based temperature proxy
               (used on platforms where sensor access is restricted)

        Memory:
            psutil.virtual_memory().used converted to GB, normalised by
            config.ram_capacity_gb.
        """
        T_norm   = 0.4   # conservative defaults if reads fail
        MEM_norm = 0.4

        # ── CPU temperature ────────────────────────────────────────────────
        try:
            cpu_temp_found = False
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                # Try known sensor keys in order of reliability
                for key in ("coretemp", "k10temp", "cpu_thermal",
                            "acpitz", "cpu-thermal", "soc_thermal"):
                    if key in temps and temps[key]:
                        raw_temp = temps[key][0].current
                        T_norm   = float(np.clip(
                            raw_temp / self.config.cpu_temp_setpoint, 0.0, 1.0
                        ))
                        cpu_temp_found = True
                        break

            if not cpu_temp_found:
                # Fallback: normalised CPU% as a load-pressure proxy
                cpu_pct = psutil.cpu_percent(interval=None)
                T_norm  = float(np.clip(cpu_pct / 100.0, 0.0, 1.0))
        except Exception:
            pass

        # ── RAM usage ──────────────────────────────────────────────────────
        try:
            vm       = psutil.virtual_memory()
            ram_used = vm.used / (1024 ** 3)   # bytes → GB
            MEM_norm = float(np.clip(
                ram_used / self.config.ram_capacity_gb, 0.0, 1.0
            ))
        except Exception:
            pass

        return np.array([T_norm, MEM_norm], dtype=np.float32)

    # ── Simulation backend (no telemetry libs) ────────────────────────────

    def _read_simulated(self) -> np.ndarray:
        """
        Realistic hardware simulation (used when neither pynvml nor psutil
        is available):
        - Temperature: sinusoidal oscillation + Gaussian noise
        - Memory:      sawtooth (gradual fill, periodic GC)
        """
        self._sim_time += self.config.poll_interval_ms / 1000.0
        sim_temp = (
            55.0
            + 20.0 * np.sin(self._sim_time * 0.1)
            + np.random.normal(0, 1.5)
        )
        sim_mem  = (
            0.3
            + 0.4  * ((self._sim_time * 0.05) % 1.0)
            + np.random.normal(0, 0.02)
        )
        # Use effective_setpoint so PID is calibrated correctly
        T_norm   = np.clip(sim_temp / self.effective_setpoint, 0.0, 1.0)
        MEM_norm = np.clip(sim_mem,  0.0, 1.0)
        return np.array([T_norm, MEM_norm], dtype=np.float32)


# =============================================================================
# §3  PID CONTROLLER
# =============================================================================

class PIDController:
    """
    Discrete-time PID controller that maps hardware thermal error → Φ_physical.

    Works identically for GPU and CPU modes — the sidecar already normalises
    the raw sensor reading to the correct set-point before passing it in.

    Φ_physical ∈ [0, 1]:
        → 0   hardware is cool / lightly loaded → lower τ (be diligent)
        → 1   hardware is hot  / under pressure → raise  τ (be frugal)

    Anti-windup: integral term is clamped to ±integral_limit.
    """

    def __init__(self, config: SPITConfig, sidecar: "HardwareTelemetrySidecar"):
        """
        Args:
            config:  SPITConfig
            sidecar: The paired HardwareTelemetrySidecar — used to read the
                     correct mode-specific set-point for error computation.
        """
        self.Kp = config.Kp
        self.Ki = config.Ki
        self.Kd = config.Kd
        # Use effective_setpoint from sidecar (GPU temp SP or CPU temp SP)
        self.setpoint       = sidecar.effective_setpoint
        self.integral_limit = config.integral_limit

        self._integral:   float = 0.0
        self._prev_error: float = 0.0
        self._prev_time:  float = time.time()

    def step(self, T_norm: float, MEM_norm: float) -> float:
        """
        Compute Φ_physical from the normalised health vector.

        Args:
            T_norm:   h[0] from sidecar — normalised temperature [0, 1]
                      (GPU temp / GPU setpoint  OR  CPU temp / CPU setpoint)
            MEM_norm: h[1] from sidecar — normalised memory pressure [0, 1]
                      (VRAM used / VRAM capacity  OR  RAM used / RAM capacity)

        Returns:
            float in [0, 1] — Resource Constraint Modifier Φ_physical
        """
        now = time.time()
        dt  = max(now - self._prev_time, 1e-3)

        # Convert normalised temperature to absolute (°C) for error term
        T_celsius = T_norm * self.setpoint
        error     = T_celsius - self.setpoint

        P = self.Kp * error
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        I = self.Ki * self._integral
        D = self.Kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = now

        # Memory pressure adds a secondary 10% weight to Φ
        mem_pressure = 0.10 * MEM_norm
        phi = P + I + D + mem_pressure
        return float(np.clip(phi, 0.0, 1.0))


# =============================================================================
# §4  SOCIO-IMPACT VECTOR  Ψ_socio
# =============================================================================

def compute_psi_socio(
    retweet_count:   float,
    follower_count:  float,
    burst_rate_ppm:  float,
    platform:        PlatformType,
    config:          SPITConfig,
) -> float:
    """
    Compute Ψ_socio — the Socio-Impact Multiplier.

    Ψ_socio ∈ [0, 1]:
        → 0   low social urgency  → threshold stays high (DistilBERT suffices)
        → 1   high social urgency → threshold actively lowered (escalate)

    Variables:
        V_vir  (virality)   = normalised retweet count       ∈ [0,1]
        A_auth (authority)  = log-normalised follower count   ∈ [0,1]
        B_burst (burst)     = normalised posting velocity     ∈ [0,1]
        H_socio (entropy)   = platform-level sarcasm prior    ∈ [0,1]

    Weights (justified by misclassification cost research):
        w_vir   = 0.30   Viral posts have cascading reach
        w_auth  = 0.30   Influencer misclassification risk
        w_burst = 0.25   Burst detection enables proactive frugality
        w_plat  = 0.15   Platform entropy baseline

    Args:
        retweet_count:  Raw retweet/share count
        follower_count: Author's follower count
        burst_rate_ppm: Current hashtag/topic posts-per-minute
        platform:       PlatformType enum
        config:         SPITConfig

    Returns:
        float in [0, 1]
    """
    V_vir  = np.clip(retweet_count / config.retweet_cap, 0.0, 1.0)

    if follower_count > 0:
        A_auth = np.clip(
            np.log1p(follower_count) / np.log1p(config.follower_cap),
            0.0, 1.0,
        )
    else:
        A_auth = 0.0

    B_burst = np.clip(burst_rate_ppm / config.burst_cap, 0.0, 1.0)
    H_socio = platform.value

    w = np.array([0.30, 0.30, 0.25, 0.15], dtype=np.float32)
    s = np.array([V_vir, A_auth, B_burst, H_socio], dtype=np.float32)

    psi = float(np.dot(w, s))
    return float(np.clip(psi, 0.0, 1.0))


# =============================================================================
# §5  LINGUISTIC ENTROPY  H_token
# =============================================================================

def compute_h_token(token_ids: list, vocab_size: int = 30522) -> float:
    """
    Compute normalised token-level linguistic entropy (H_token).

    H_token captures lexical diversity / complexity of the input text.
    High entropy → unusual vocabulary → potentially ambiguous sentiment
    → lower threshold warranted (favour escalation).

    Normalised by log(vocab_size) so H_token ∈ [0, 1].

    Args:
        token_ids:  List of integer token IDs from DistilBERT tokeniser
        vocab_size: Vocabulary size (DistilBERT default = 30,522)

    Returns:
        float in [0, 1]
    """
    if not token_ids:
        return 0.0
    tokens = np.array(token_ids, dtype=np.int32)
    _, counts = np.unique(tokens, return_counts=True)
    probs      = counts / counts.sum()
    entropy    = -np.sum(probs * np.log(probs + 1e-12))
    normalised = entropy / np.log(vocab_size + 1e-12)
    return float(np.clip(normalised, 0.0, 1.0))


# =============================================================================
# §6  THE SPIT FUNCTION  τ_dynamic
# =============================================================================

def spit(
    phi_physical:    float,
    retweet_count:   float,
    follower_count:  float,
    burst_rate_ppm:  float,
    platform:        PlatformType,
    token_ids:       list,
    config:          SPITConfig = SPITConfig(),
) -> dict:
    """
    Socio-Physical Integrated Thresholding (SPIT) Function.

    Computes the dynamic escalation threshold τ_dynamic that governs whether
    a post is processed by DistilBERT (Tier-1) alone or escalated to BERT (Tier-2).

    Formal definition:
        τ_dynamic = σ(Φ_physical·(1−Ψ_socio) + H_token)·(τ_max−τ_min) + τ_min

    Intuition:
        - Φ_physical ↑  → hardware under pressure → push τ UP (be frugal)
        - Ψ_socio ↑     → post is socially urgent → push τ DOWN (escalate)
        - H_token ↑     → text is linguistically complex → push τ DOWN (escalate)

    Returns:
        dict with τ_dynamic and all intermediate components for logging/ablation.
    """
    psi_socio = compute_psi_socio(
        retweet_count, follower_count, burst_rate_ppm, platform, config
    )
    h_token = compute_h_token(token_ids)

    physical_term = phi_physical * (1.0 - psi_socio)
    z             = physical_term + h_token
    sigma_z       = 1.0 / (1.0 + np.exp(-z))
    tau_dynamic   = sigma_z * (config.tau_max - config.tau_min) + config.tau_min
    tau_dynamic   = float(np.clip(tau_dynamic, config.tau_min, config.tau_max))

    return {
        "tau_dynamic":   tau_dynamic,
        "phi_physical":  phi_physical,
        "psi_socio":     psi_socio,
        "h_token":       h_token,
        "physical_term": physical_term,
        "sigmoid_input": z,
    }


# =============================================================================
# §7  DATA STRUCTURES & CASCADE ROUTER
# =============================================================================

@dataclass
class PostContext:
    """All metadata required to make a SPIT routing decision for one post."""
    text:            str
    token_ids:       list
    distilbert_conf: float
    retweet_count:   float        = 0.0
    follower_count:  float        = 0.0
    burst_rate_ppm:  float        = 0.0
    platform:        PlatformType = PlatformType.UNKNOWN


@dataclass
class RoutingDecision:
    """Output of the cascade router for a single post."""
    escalate:        bool
    tau_dynamic:     float
    distilbert_conf: float
    spit_components: dict
    tier:            str = field(init=False)

    def __post_init__(self):
        self.tier = "BERT (Tier-2)" if self.escalate else "DistilBERT (Tier-1)"


class SPITCascadeRouter:
    """
    Orchestrates the DistilBERT → BERT cascade using SPIT thresholding.

    Automatically detects compute mode (GPU / CPU / SIM) and configures
    telemetry accordingly — no manual setup required.

    Usage:
        router = SPITCascadeRouter()   # auto-detects GPU or CPU
        router.start()
        decision = router.route(post_context)
        router.stop()
    """

    def __init__(self, config: SPITConfig = SPITConfig()):
        self.config  = config
        self.sidecar = HardwareTelemetrySidecar(config)
        # PID receives sidecar so it can read the correct set-point
        self.pid     = PIDController(config, self.sidecar)
        self._started = False

    def start(self):
        """Start the hardware telemetry sidecar thread."""
        self.sidecar.start()
        self._started = True
        print(f"[SPIT] Cascade router started | compute mode: {self.sidecar.mode.value.upper()}")

    def stop(self):
        """Shut down the sidecar thread cleanly."""
        self.sidecar.stop()
        print("[SPIT] Cascade router stopped.")

    @property
    def compute_mode(self) -> ComputeMode:
        """Expose the detected compute mode for upstream logging."""
        return self.sidecar.mode

    def route(self, post: PostContext) -> RoutingDecision:
        """
        Make a SPIT-governed routing decision for one post.

        Steps:
            1. Read hardware health vector h from sidecar (O(1), lock-protected)
               h[0] = normalised temperature  (GPU or CPU, same scale)
               h[1] = normalised memory usage (VRAM or RAM, same scale)
            2. Step PID → Φ_physical
            3. Call spit() → τ_dynamic
            4. Escalate to BERT iff DistilBERT confidence < τ_dynamic

        Args:
            post: PostContext — text, token_ids, distilbert_conf, social metadata.

        Returns:
            RoutingDecision
        """
        if not self._started:
            raise RuntimeError("Call router.start() before routing posts.")

        h        = self.sidecar.read()   # [T_norm, MEM_norm]
        T_norm   = float(h[0])
        MEM_norm = float(h[1])

        # PID step — works for both GPU and CPU because everything is normalised
        phi = self.pid.step(T_norm, MEM_norm)

        result = spit(
            phi_physical   = phi,
            retweet_count  = post.retweet_count,
            follower_count = post.follower_count,
            burst_rate_ppm = post.burst_rate_ppm,
            platform       = post.platform,
            token_ids      = post.token_ids,
            config         = self.config,
        )

        escalate = post.distilbert_conf < result["tau_dynamic"]

        return RoutingDecision(
            escalate        = escalate,
            tau_dynamic     = result["tau_dynamic"],
            distilbert_conf = post.distilbert_conf,
            spit_components = result,
        )
