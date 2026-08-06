#!/usr/bin/env python3
"""
collect_benchmarks.py — ConfigProfiler, Step 1: real data collection.

ADAPTED FOR QWEN3 on August 3, 2026 (Day 2). See EXECUTION_LOG.md for the
full detail of what changed compared to the original generic version.

WHAT IS REUSED UNCHANGED (validated on a real Pixel 7a, Day 1):
    - the entire Device class: battery_pct, battery_watts, cpu_temp_c (via
      dumpsys thermalservice, documented SELinux workaround),
      max_freq_drop_pct. Do not touch without re-justifying in writing.

WHAT WAS REWRITTEN (Qwen3-specific, absent from the generic version):
    - run_on_device_benchmark(): actually calls `llama_main` (cross-compiled
      ARM64 runner, validated Day 2 — see EXECUTION_LOG.md, results
      73.45/22.00 tok/s over USB, 103.17/36.81 tok/s over Wi-Fi/discharging),
      with the Qwen3 chat template (<|im_start|>...) and parsing of the
      PyTorchObserver JSON actually emitted by the runner (more reliable
      than the old generic tokens_per_sec: regex, which never matched any
      real ExecuTorch runner output).
    - Power measurement: FIXED on Day 2 after a methodological flaw was
      caught on the first real run (power abnormally low during inference
      vs. at rest — see EXECUTION_LOG.md). The old version called
      battery_watts() once, AFTER inference finished, which measured an
      already-settled state, not the real draw during the run. Fixed with
      parallel sampling (a dedicated thread, ~every 0.4s) DURING the entire
      blocking llama_main call. The CSV now reports
      watts_mean/watts_min/watts_max/watts_n_samples instead of a single
      post-hoc value.

METHODOLOGICAL DEEPENING added Day 2 (beyond the power fix above), in
response to "how do we go further for a demanding jury":
    1. Idle baseline (--idle-baseline-s, default 2s): measures resting
       power draw (screen off, no inference) right before each run.
       watts_delta_mean = watts_mean - baseline_watts_mean gives the cost
       attributable to inference itself, not a raw value with no reference.
    2. Screen-state control (Device.ensure_screen_off): a confounding
       variable that was uncontrolled until now — a small quantized model
       can draw less power than the screen itself. Forced off before every
       baseline measurement.
    3. cache_state dimension (cold/warm) via --inter-rep-pause-s: the gap
       observed between isolated runs (Day 1-2, 22-37 tok/s) and back-to-
       back runs (Day 2, ~48 tok/s) is no longer an unexplained anomaly but
       an explicit experimental variable. 0 = historical behavior
       (back-to-back runs, cache_state="warm"), >0 = pause before each rep
       to induce a "cold" state.
    4. --check-throttle-access: a diagnostic to run BEFORE any real
       collection to know whether reading scaling_cur_freq/cpuinfo_max_freq
       is blocked by SELinux (like thermal_zone) — avoids presenting
       `throttled` as more reliable than it actually is if this read fails
       silently and falls back to 0.0.

⚠️ TODO BEFORE THE REAL COLLECTION — one remaining item:
    1. Quantization -> .pte file mapping: as of now, only one .pte exists
       on the device (qwen3_0_6b.pte, config qwen3_xnnpack_q8da4w). A real
       int8/int4 matrix requires exporting and pushing a separate .pte per
       quantization level before running the full collection.

DECISION SETTLED (no longer a TODO): Qwen3's "thinking" mode is disabled
via the soft switch "/no_think" appended to every user prompt. Reason: it
pollutes the demo video (sequence C) and skews the token/latency metrics of
the jargon-fidelity dataset without adding value to the project (we measure
translation fidelity, not the quality of internal reasoning). See
EXECUTION_LOG.md for the full discussion.

No external dependencies: Python 3.8+, adb in PATH, USB or wireless
debugging enabled.

⚠️ ABOUT --dry-run — READ BEFORE TRUSTING ANY NUMBER FROM THIS SCRIPT:
    This script has a --dry-run mode that simulates adb responses (fake
    battery level, fake thermal readings, fake watts, fake PyTorchObserver
    JSON with randomized tok/s) so the collection PIPELINE (tier waiting,
    warmup, resume logic, CSV schema) can be tested without a connected
    device. --dry-run output is entirely synthetic and MUST NEVER be cited,
    published, or presented as a real measurement — it exists solely to
    validate that the script's logic works before running it on real
    hardware. Every number in this project's README, pitch, or video comes
    from a run WITHOUT --dry-run, on a physical Pixel 7a. If you are
    reading this code and see a result that looks suspiciously clean or
    randomized, check whether --dry-run was used to produce it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Configuration matrix
# --------------------------------------------------------------------------
#
# Deliberately reduced compared to the original generic version ("depth over
# breadth", a decision made upfront to keep a solo 13-day timeline on track
# — see the Day 1 planning discussion). "mixed" quantization removed: not
# enough added value for a solo dev's time budget. Threads limited to 2/4:
# see TODO #1 above before actually enabling more.

QUANTIZATIONS = ["q8da4w"]   # TODO #2: add "int4" once that .pte is exported and pushed
THREADS = [2, 4]

# Path to the .pte on the DEVICE (not local) for each quantization level.
# To be filled in as more exports happen (see TODO #2).
QUANT_TO_PTE = {
    "q8da4w": "qwen3_0_6b.pte",
}

# (tier_id, user-facing_label, verification_predicate)
# Order high->mid->low (Day 3 re-collection, device already charged to 81%
# after the throttling-bug fix): follows the natural discharge trajectory
# from the phone's current state, symmetric to the low->mid->high order used
# for the first collection (device then at 16%). Order has no impact on
# measurement validity, only on how efficient the collection session is.
BATTERY_TIERS = [
    ("high", ">70%", lambda pct: pct > 70),
    ("mid", "30-50%", lambda pct: 30 <= pct <= 50),
    ("low", "<20%", lambda pct: pct < 20),
]

THERMAL_CONDITIONS = [
    ("ambient", "normal ambient temperature, no preheating"),
    ("preheated", "phone preheated for 15-20 min (sunlight/heat source), "
                  "run starts at >=40°C on the CPU thermal zone"),
]

REPS_PER_CONFIG = 3          # repetitions per (quant, threads, battery, thermal)
WARMUP_RUNS = 1              # discarded runs before real measurements, per block
PREHEAT_MIN_TEMP_C = 40.0

# Folder on the DEVICE where model/tokenizer/runner are already deployed (Day 1-2)
DEVICE_DIR = "/data/local/tmp/configprofiler"
TOKENIZER_FILENAME = "tokenizer.json"
RUNNER_FILENAME = "llama_main"

# Test prompt — a single one for now (pure performance measurement, no
# jargon fidelity here — that's a separate dataset, section 5 of the
# reference document, not yet tackled)
DEFAULT_PROMPT = "Who is the president of the US?"

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "configprofiler_dataset.csv"

CSV_FIELDS = [
    "run_id", "timestamp_iso", "quantization", "threads",
    "battery_tier", "battery_pct_actual", "battery_saver_active",
    "thermal_condition",
    "thermal_zone_used", "cpu_temp_start_c", "cpu_temp_end_c",
    "throttled", "freq_drop_pct", "prefill_tokens_per_sec",
    "decode_tokens_per_sec", "prompt_tokens", "generated_tokens",
    "model_load_ms", "prefill_duration_ms", "decode_duration_ms",
    "watts_mean", "watts_min", "watts_max",
    "watts_n_samples", "watts_samples_json", "baseline_watts_mean",
    "watts_delta_mean", "cache_state", "swap_latency_ms", "is_warmup", "notes",
]

# --------------------------------------------------------------------------
# ADB / device layer — UNCHANGED from the generic version, validated on a
# real Pixel 7a Day 1 (SELinux/thermal fix and USB/watts bias already
# documented and tested, see README-collecte.md). Do not modify without
# revalidating on a real device and documenting the change in
# EXECUTION_LOG.md.
# --------------------------------------------------------------------------

class AdbError(RuntimeError):
    pass


class Device:
    """Thin wrapper around adb shell. No external dependency."""

    def __init__(self, serial: Optional[str] = None, dry_run: bool = False):
        self.serial = serial
        self.dry_run = dry_run
        self._thermal_zone_cache: Optional[str] = None
        self._current_sign_cache: Optional[int] = None  # +1 or -1
        self._mock_temp_c: float = 36.0  # rises on each read in dry-run to simulate preheating
        self._cpufreq_unavailable: bool = False

    def _adb_base(self) -> list[str]:
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def shell(self, command: str, timeout: float = 30.0) -> str:
        if self.dry_run:
            return self._mock_shell(command)
        cmd = self._adb_base() + ["shell", command]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise AdbError(f"Timeout on: {command}") from e
        if result.returncode != 0 and not result.stdout:
            raise AdbError(f"adb shell failed ({command}): {result.stderr.strip()}")
        return result.stdout

    def _mock_shell(self, command: str) -> str:
        """Simulated responses for testing the script without a connected device (--dry-run)."""
        if "dumpsys battery" in command:
            return "  level: 55\n"
        if "dumpsys thermalservice" in command:
            self._mock_temp_c = min(self._mock_temp_c + 1.5, 46.0)
            base = self._mock_temp_c
            return (
                "Current temperatures from HAL:\n"
                f"        Temperature{{mValue={base:.4f}, mType=9, mName=TPU, mStatus=0}}\n"
                f"        Temperature{{mValue={base:.4f}, mType=1, mName=G3D, mStatus=0}}\n"
                f"        Temperature{{mValue={base + 1.0:.4f}, mType=0, mName=BIG, mStatus=0}}\n"
                f"        Temperature{{mValue={base:.4f}, mType=0, mName=LITTLE, mStatus=0}}\n"
                f"        Temperature{{mValue={base:.4f}, mType=0, mName=MID, mStatus=0}}\n"
                f"        Temperature{{mValue={base - 3.0:.4f}, mType=2, mName=battery, mStatus=0}}\n"
                "Current cooling devices from HAL:\n"
            )
        if "current_now" in command and "voltage_now" in command:
            # Combined command (Day 2, sampling optimization):
            # `cat f1 f2` concatenates both files, one value per line.
            return "-850000\n3900000\n"
        if "current_now" in command:
            return "-850000\n"
        if "voltage_now" in command:
            return "3900000\n"
        if "scaling_cur_freq" in command and "cpuinfo_max_freq" in command:
            # Simulates 8 big.LITTLE cores: 4 LITTLE (cur close to max,
            # ~1.8GHz) + 4 BIG (idle, low cur vs max ~2.85GHz) — to test
            # that max_freq_drop_pct correctly selects the MOST ACTIVE core
            # (LITTLE here) rather than mixing clusters.
            lines = []
            for _ in range(4):
                lines.append("1700000 1800000")  # LITTLE, close to its max
            for _ in range(4):
                lines.append("300000 2850000")   # BIG, essentially idle
            return "\n".join(lines) + "\n"
        if "dumpsys power" in command:
            return "  mWakefulness=Awake\n"
        if "input keyevent" in command:
            return ""
        if "low_power" in command:
            return "0\n"
        if RUNNER_FILENAME in command:
            # simulates a plausible PyTorchObserver line
            prefill = round(random.uniform(50.0, 110.0), 2)
            decode = round(random.uniform(15.0, 40.0), 2)
            load_end = random.randint(1000, 2500)
            inf_start = load_end + 50
            eval_end = inf_start + random.randint(150, 400)
            inf_end = eval_end + random.randint(2000, 4000)
            payload = {
                "prefill_token_per_sec": prefill,
                "decode_token_per_sec": decode,
                "prompt_tokens": 13,
                "generated_tokens": random.randint(80, 128),
                "model_load_start_ms": 0,
                "model_load_end_ms": load_end,
                "inference_start_ms": inf_start,
                "prompt_eval_end_ms": eval_end,
                "inference_end_ms": inf_end,
            }
            return f"PyTorchObserver {json.dumps(payload)}\n"
        return ""

    # ---- Battery ----

    def battery_pct(self) -> int:
        out = self.shell("dumpsys battery")
        m = re.search(r"level:\s*(\d+)", out)
        if not m:
            raise AdbError("Could not read battery level via dumpsys battery")
        return int(m.group(1))

    def battery_watts(self) -> float:
        """Watt estimate from current_now (µA) and voltage_now (µV).

        WARNING: the sign of current_now depends on the firmware (positive
        or negative while discharging, depending on the OEM). We calibrate
        the sign on the first call and freeze it for the rest of the
        session — verify consistency with a manual test (unplug the
        charger, check the displayed value makes sense).

        DAY 2 OPTIMIZATION: both files are read in a single adb call
        (`cat f1 f2`) rather than two sequential calls. Discovered while
        digging into the watts_samples_json time series: two adb calls per
        sample meant the real interval between samples (~1.0-1.4s) far
        exceeded the WattsSampler's interval_s=0.4 parameter — see
        EXECUTION_LOG.md. A single adb call roughly doubles the achieved
        temporal resolution.
        """
        out = self.shell(
            "cat /sys/class/power_supply/battery/current_now "
            "/sys/class/power_supply/battery/voltage_now"
        ).strip()
        lines = [l for l in out.splitlines() if l.strip()]
        if len(lines) < 2:
            raise AdbError(
                f"Unexpected output for combined current_now/voltage_now "
                f"(expected 2 lines, got {len(lines)}): {out!r}"
            )
        cur_raw, volt_raw = lines[0].strip(), lines[1].strip()
        current_ua = float(cur_raw)
        voltage_uv = float(volt_raw)
        if self._current_sign_cache is None:
            self._current_sign_cache = -1 if current_ua < 0 else 1
        current_a = abs(current_ua) / 1e6
        voltage_v = abs(voltage_uv) / 1e6
        return round(current_a * voltage_v, 4)

    # ---- Thermal ----
    #
    # IMPORTANT NOTE (discovered during real data collection on a Pixel 7a,
    # not a hypothesis): direct access to /sys/class/thermal/thermal_zone*/
    # temp is blocked by SELinux for the "shell" user (the one used by adb),
    # even without root — documented behavior on recent Pixels, not a bug in
    # this script. The source used instead is `dumpsys thermalservice`, an
    # Android service accessible without root via adb shell.
    #
    # On the Tensor G2 (Pixel 7a), this service exposes the three CPU
    # clusters under the names LITTLE / MID / BIG (mType=0 = CPU in the
    # Android ThermalHAL enum), in the "Current temperatures from HAL"
    # section (NOT "Cached temperatures", which holds stale/stable values
    # not representative of the current state — verified against real
    # output).

    _THERMAL_HAL_CPU_TYPE = "0"  # mType=0 == CPU in the Android ThermalHAL enum

    def _parse_thermal_hal_cpu(self) -> dict[str, float]:
        """Extracts CPU cluster temperatures from the 'Current temperatures
        from HAL' section of dumpsys thermalservice."""
        out = self.shell("dumpsys thermalservice")
        if "Current temperatures from HAL" not in out:
            raise AdbError(
                "'Current temperatures from HAL' section missing from "
                "dumpsys thermalservice. Full output needs manual inspection."
            )
        section = out.split("Current temperatures from HAL", 1)[1]
        section = section.split("Current cooling devices", 1)[0]

        cpu_temps = {}
        pattern = re.compile(
            r"Temperature\{mValue=([\d.]+),\s*mType=(-?\d+),\s*mName=([\w\-]+)"
        )
        for value_str, mtype, name in pattern.findall(section):
            if mtype == self._THERMAL_HAL_CPU_TYPE:
                cpu_temps[name] = float(value_str)

        if not cpu_temps:
            raise AdbError(
                "No mType=0 (CPU) entry found in 'Current temperatures "
                "from HAL'. Manually inspect 'adb shell dumpsys "
                "thermalservice' and adapt _THERMAL_HAL_CPU_TYPE / the "
                "parsing if your device names its clusters differently."
            )
        return cpu_temps

    def cpu_temp_c(self) -> float:
        """Returns the temperature of the hottest CPU cluster (most
        relevant for detecting imminent throttling)."""
        cpu_temps = self._parse_thermal_hal_cpu()
        if self._thermal_zone_cache is None:
            self._thermal_zone_cache = ", ".join(sorted(cpu_temps.keys()))
            print(f"[device] CPU clusters detected via dumpsys thermalservice: "
                  f"{self._thermal_zone_cache}", file=sys.stderr)
        return max(cpu_temps.values())

    def thermal_zone_used(self) -> str:
        if self._thermal_zone_cache is None:
            cpu_temps = self._parse_thermal_hal_cpu()
            self._thermal_zone_cache = ", ".join(sorted(cpu_temps.keys()))
        return f"dumpsys thermalservice (HAL, mType=0): {self._thermal_zone_cache}"

    # ---- Battery Saver — added Day 3: a hidden variable that was
    # uncontrolled until now. On Pixel, the default auto-enable threshold
    # for Battery Saver is 20% — exactly the boundary of this script's "low"
    # tier (<20%). If active, it throttles the CPU and kills background
    # activity: measuring "low battery" without knowing whether this mode is
    # active would mix two different effects under a single label.

    def is_battery_saver_active(self) -> bool:
        out = self.shell("settings get global low_power")
        return out.strip() == "1"

    # ---- CPU freq / throttling ----

    def _cpu_freqs(self) -> list[tuple[int, int]]:
        """Returns a list of (current_freq, max_freq) pairs PER CORE, in the
        same order for both readings — essential on a big.LITTLE
        architecture like the Tensor G2, where LITTLE cores (~1.8GHz max)
        and BIG cores (~2.85GHz max) have different ceilings.

        DAY 3 FIX (see EXECUTION_LOG.md): the previous version returned two
        separate lists and compared global max(cur) to global max(mx) — a
        mix across different cores that produced an artificially high
        "freq_drop_pct" (23-44%, throttled=True on 100% of the reference
        collection's runs) even with no real overheating, simply because
        the active LITTLE core (max ~1.8GHz) was being compared against the
        idle BIG core's ceiling (max ~2.85GHz).

        NOTE: on this project, direct access to /sys/class/thermal was found
        to be blocked by SELinux for the shell user on the Pixel 7a (see the
        Thermal section above) — /sys/devices/system/cpu/... might be
        blocked too depending on the Android version. If so, this method
        returns an empty list rather than crashing the whole collection;
        freq_drop_pct will then be 0.0 for every run and throttled will be
        based solely on the temperature drop observed during the run (less
        reliable) — verify manually with the command below before running a
        real collection session.
        """
        if self._cpufreq_unavailable:
            return []
        try:
            # Paired reading, core by core, in a single shell command (one
            # "cur max" line per core), to guarantee index correspondence
            # between the two values — not two separate reads.
            paired_out = self.shell(
                "for cpu in /sys/devices/system/cpu/cpu*/cpufreq; do "
                "c=$(cat $cpu/scaling_cur_freq 2>/dev/null); "
                "m=$(cat $cpu/cpuinfo_max_freq 2>/dev/null); "
                "if [ -n \"$c\" ] && [ -n \"$m\" ]; then echo \"$c $m\"; fi; "
                "done"
            )
        except AdbError:
            paired_out = ""
        pairs = []
        for line in paired_out.splitlines():
            parts = line.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                pairs.append((int(parts[0]), int(parts[1])))
        if not pairs:
            self._cpufreq_unavailable = True
            print("[device] WARNING: paired scaling_cur_freq/cpuinfo_max_freq "
                  "read empty or refused. Verify manually with: "
                  "adb shell 'for cpu in /sys/devices/system/cpu/cpu*/cpufreq; "
                  "do echo $cpu: $(cat $cpu/scaling_cur_freq) / "
                  "$(cat $cpu/cpuinfo_max_freq); done' — if blocked by SELinux "
                  "like thermal_zone, throttled will be estimated solely from "
                  "the temperature drop during the run, not the real CPU "
                  "frequency.", file=sys.stderr)
        return pairs

    def max_freq_drop_pct(self) -> float:
        """% drop between the theoretical max frequency and the current
        frequency, computed on the MOST ACTIVE core at call time (the one
        with the highest current frequency) — not a mix across different
        clusters' cores. Positive = real slowdown of that core relative to
        ITS OWN ceiling, not another cluster's."""
        pairs = self._cpu_freqs()
        if not pairs:
            return 0.0
        # Most active core = the one with the highest current frequency —
        # it's ITS ceiling we compare against, not another cluster's.
        busiest_cur, busiest_max = max(pairs, key=lambda p: p[0])
        if busiest_max == 0:
            return 0.0
        return round(100.0 * (1 - busiest_cur / busiest_max), 2)

    # ---- Screen — added Day 2: a confounding variable that was
    # uncontrolled until now. A small quantized model can draw less power
    # than the screen itself, so an uncontrolled screen state between runs
    # would silently invalidate watts comparisons.

    def is_screen_on(self) -> bool:
        out = self.shell("dumpsys power")
        return "mWakefulness=Awake" in out

    def ensure_screen_off(self) -> None:
        """Forces the screen off if it's on. Idempotent — does nothing if
        already off (avoids accidentally turning it back on with a blind
        toggle)."""
        if self.is_screen_on():
            self.shell("input keyevent 26")  # power key, toggles
            time.sleep(0.5)


# --------------------------------------------------------------------------
# Watts sampling DURING inference — Day 2 methodological fix (see the
# module docstring above and EXECUTION_LOG.md).
#
# The old approach called device.battery_watts() once, after the run ended
# — which measures an already-settled state, not the real draw during
# inference. This sampler runs in a separate thread for the entire duration
# of the blocking llama_main call, via independent adb calls (adb
# multiplexes several shell sessions without conflict).
# --------------------------------------------------------------------------

class WattsSampler:
    """Continuously samples device.battery_watts() in a thread, until
    .stop() is called. Designed to bracket a blocking call (e.g. llama_main
    inference) and measure power draw DURING the run, not just before/after.

    Each sample is timestamped in seconds relative to .start() (HOST/WSL
    clock, not the device's) — allows reconstructing a power/time curve per
    run for later analysis. ⚠️ This timestamp is NOT synchronized with the
    PyTorchObserver JSON's timestamps (DEVICE clock) — see the limitation
    documented in EXECUTION_LOG.md before attempting to correlate a watts
    sample with a precise phase (prefill/decode) of the run.
    """

    def __init__(self, device: Device, interval_s: float = 0.4):
        self.device = device
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []  # (elapsed_s, watts)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._t_start: Optional[float] = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                w = self.device.battery_watts()
                elapsed = time.perf_counter() - self._t_start
                with self._lock:
                    self.samples.append((round(elapsed, 3), w))
            except AdbError:
                # A failed read during sampling shouldn't crash the whole
                # run — just skip it.
                pass
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self._stop_event.clear()
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> "WattsStats":
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            samples = list(self.samples)
        if not samples:
            # Safety net: run too short for even one sample (unlikely with
            # interval_s=0.4s and an inference of several seconds, but we
            # don't want to crash on it).
            return WattsStats(mean=float("nan"), min=float("nan"),
                               max=float("nan"), n_samples=0, raw_samples=[])
        watts_values = [w for _, w in samples]
        return WattsStats(
            mean=round(sum(watts_values) / len(watts_values), 4),
            min=round(min(watts_values), 4),
            max=round(max(watts_values), 4),
            n_samples=len(watts_values),
            raw_samples=samples,
        )


@dataclasses.dataclass
class WattsStats:
    mean: float
    min: float
    max: float
    n_samples: int
    raw_samples: list = dataclasses.field(default_factory=list)  # [(elapsed_s, watts), ...]


def measure_idle_baseline(device: Device, duration_s: float, interval_s: float = 0.4) -> WattsStats:
    """Measures resting power draw (screen off, no inference) right before a
    run, to compute a watts_inference - watts_idle delta rather than
    presenting a raw value with no point of comparison. Added Day 2 following
    the remark on going deeper methodologically (see EXECUTION_LOG.md).

    duration_s=0 disables the measurement (returns NaN) — useful to keep
    runs fast during script development/debugging.
    """
    if duration_s <= 0:
        return WattsStats(mean=float("nan"), min=float("nan"),
                           max=float("nan"), n_samples=0, raw_samples=[])
    device.ensure_screen_off()
    sampler = WattsSampler(device, interval_s=interval_s)
    sampler.start()
    time.sleep(duration_s)
    return sampler.stop()




@dataclasses.dataclass
class BenchmarkResult:
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    prompt_tokens: int
    generated_tokens: int
    model_load_ms: float
    prefill_duration_ms: float
    decode_duration_ms: float
    raw_output: str


def build_llama_main_command(quantization: str, threads: Optional[int], prompt: str) -> str:
    """Builds the shell command executed on the device.

    Flag confirmed via `llama_main --help` on the device (Day 2):
    -cpu_threads (int32, default -1 = automatic heuristic). gflags accepts
    both -cpu_threads=N and --cpu_threads=N (tested with --tokenizer_path in
    earlier runs) — we keep the double-dash style for consistency with the
    rest of the project.

    DAY 2 DECISION (settled, see EXECUTION_LOG.md): Qwen3's "thinking" mode
    disabled via the soft switch "/no_think" appended to the user message.
    The hard switch (enable_thinking=False) is a Python-side option on the
    HuggingFace tokenizer (apply_chat_template), unusable from this C++
    runner which builds the prompt as raw text — the textual soft switch is
    therefore the only option available here. Confirmed to work for Qwen3
    (not Qwen3-VL, not Qwen3.5, which behave differently).
    """
    if quantization not in QUANT_TO_PTE:
        raise ValueError(
            f"Quantization '{quantization}' is not mapped to a known .pte. "
            f"Available mappings: {list(QUANT_TO_PTE.keys())}. "
            f"Export and push the missing .pte before continuing (TODO #2)."
        )
    pte_filename = QUANT_TO_PTE[quantization]
    templated_prompt = f"<|im_start|>user {prompt} /no_think<|im_end|><|im_start|>assistant"

    parts = [
        f"cd {DEVICE_DIR} &&",
        f"./{RUNNER_FILENAME}",
        f"--model_path={pte_filename}",
        f"--tokenizer_path={TOKENIZER_FILENAME}",
        f'--prompt="{templated_prompt}"',
    ]
    if threads is not None:
        parts.append(f"--cpu_threads={threads}")
    return " ".join(parts)


def run_on_device_benchmark(
    device: Device, quantization: str, threads: Optional[int], prompt: str = DEFAULT_PROMPT
) -> BenchmarkResult:
    """Runs llama_main on the device and parses the PyTorchObserver JSON line
    actually emitted by the runner (confirmed Day 2, see EXECUTION_LOG.md).
    """
    cmd = build_llama_main_command(quantization, threads, prompt)
    out = device.shell(cmd, timeout=120.0)

    json_match = re.search(r"PyTorchObserver\s+(\{.*\})", out)
    if not json_match:
        raise AdbError(
            f"PyTorchObserver line not found in llama_main output. "
            f"Raw output (last 500 chars):\n{out[-500:]}"
        )
    try:
        metrics = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        raise AdbError(f"Malformed PyTorchObserver JSON: {e}\nLine: {json_match.group(1)}") from e

    model_load_ms = metrics.get("model_load_end_ms", 0) - metrics.get("model_load_start_ms", 0)

    # Real prefill/decode durations, computed solely from DEVICE timestamps
    # (same clock on both sides of the calculation, hence reliable — unlike
    # any attempt to correlate this with the host-side watts samples, whose
    # clock is not synchronized with the device's. See EXECUTION_LOG.md for
    # the documented limitation on this point.
    inference_start = metrics.get("inference_start_ms")
    inference_end = metrics.get("inference_end_ms")
    prompt_eval_end = metrics.get("prompt_eval_end_ms")
    if inference_start is not None and prompt_eval_end is not None:
        prefill_duration_ms = float(prompt_eval_end - inference_start)
    else:
        prefill_duration_ms = float("nan")
    if prompt_eval_end is not None and inference_end is not None:
        decode_duration_ms = float(inference_end - prompt_eval_end)
    else:
        decode_duration_ms = float("nan")

    return BenchmarkResult(
        prefill_tokens_per_sec=metrics["prefill_token_per_sec"],
        decode_tokens_per_sec=metrics["decode_token_per_sec"],
        prompt_tokens=metrics["prompt_tokens"],
        generated_tokens=metrics["generated_tokens"],
        model_load_ms=float(model_load_ms),
        prefill_duration_ms=prefill_duration_ms,
        decode_duration_ms=decode_duration_ms,
        raw_output=out,
    )


# --------------------------------------------------------------------------
# Orchestration — UNCHANGED (tier/warmup/resume logic already solid)
# --------------------------------------------------------------------------

def load_completed_keys(csv_path: Path) -> set[tuple]:
    """Allows resuming an interrupted collection without duplicating runs."""
    completed = set()
    if not csv_path.exists():
        return completed
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_warmup") == "True":
                continue
            key = (
                row["quantization"], row["threads"], row["battery_tier"],
                row["thermal_condition"], row["run_id"].rsplit("-rep", 1)[-1],
            )
            completed.add(key)
    return completed


def wait_for_battery_condition(device: Device, tier_id: str, tier_label: str, check_fn) -> int:
    while True:
        pct = device.battery_pct()
        if check_fn(pct):
            print(f"[ok] Battery at {pct}% — matches tier '{tier_id}' ({tier_label}).")
            return pct
        print(f"[waiting] Current battery: {pct}% — need {tier_label} for tier "
              f"'{tier_id}'. Adjust charging (plug/unplug) and press Enter "
              f"to re-check (or type 'force' to proceed anyway).")
        answer = input("> ").strip().lower()
        if answer == "force":
            print(f"[forced] Proceeding with battery at {pct}%, outside the "
                  f"theoretical tier — will be noted in the run's notes.")
            return pct


def wait_for_thermal_condition(device: Device, condition_id: str, condition_label: str) -> None:
    if condition_id != "preheated":
        input(f"[setup] Thermal condition '{condition_id}' ({condition_label}). "
              f"Ensure a stable ambient state, then press Enter to continue.")
        return
    while True:
        temp = device.cpu_temp_c()
        if temp >= PREHEAT_MIN_TEMP_C:
            print(f"[ok] Starting CPU temperature: {temp:.1f}°C (threshold: {PREHEAT_MIN_TEMP_C}°C).")
            return
        print(f"[waiting] Current CPU temperature: {temp:.1f}°C, required threshold: "
              f"{PREHEAT_MIN_TEMP_C}°C. Keep preheating, then press Enter to re-check.")
        input("> ")


def append_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_single(
    device: Device, quantization: str, threads: Optional[int],
    battery_tier: str, battery_pct_actual: int,
    thermal_condition: str, is_warmup: bool, rep_idx: int,
    idle_baseline_s: float = 0.0, cache_state: str = "warm",
) -> dict:
    temp_start = device.cpu_temp_c()
    freq_drop_before = device.max_freq_drop_pct()

    battery_saver_active = device.is_battery_saver_active()
    if battery_saver_active:
        print(f"[⚠️ WARNING] Battery Saver mode ACTIVE during this run "
              f"({battery_tier}/{thermal_condition}/rep{rep_idx}). Frequency/"
              f"performance measurements include this mode's effect, not "
              f"just the battery level's — recorded in battery_saver_active. "
              f"Disable it manually if you want to isolate the battery "
              f"effect alone.",
              file=sys.stderr)

    # Idle baseline (screen off, no inference) — added Day 2. Measured right
    # before inference to give watts_mean a real point of comparison during
    # the run (delta = cost attributable to inference itself, not a raw
    # value with no reference). idle_baseline_s=0 disables it.
    baseline_stats = measure_idle_baseline(device, duration_s=idle_baseline_s)

    # Watts sampling DURING inference (Day 2 fix): the sampler runs in a
    # separate thread, precisely bracketing the blocking llama_main call —
    # not an isolated before/after-only measurement.
    sampler = WattsSampler(device)
    sampler.start()
    t0 = time.perf_counter()
    try:
        bench = run_on_device_benchmark(device, quantization, threads)
    finally:
        elapsed = time.perf_counter() - t0
        watts_stats = sampler.stop()

    temp_end = device.cpu_temp_c()
    freq_drop_after = device.max_freq_drop_pct()

    throttled = freq_drop_after > 15.0 or freq_drop_after > freq_drop_before + 10.0

    watts_delta_mean = (
        round(watts_stats.mean - baseline_stats.mean, 4)
        if watts_stats.n_samples > 0 and baseline_stats.n_samples > 0
        else float("nan")
    )

    run_id = f"{quantization}-t{threads}-{battery_tier}-{thermal_condition}-{cache_state}-rep{rep_idx}"
    notes = f"wall_time_s={elapsed:.2f}"
    if watts_stats.n_samples == 0:
        notes += "; WARNING: no watts sample collected during this run (see watts_n_samples)"
    if idle_baseline_s <= 0:
        notes += "; idle baseline disabled (idle_baseline_s=0), watts_delta_mean not computable"

    return {
        "run_id": run_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "quantization": quantization,
        "threads": threads,
        "battery_tier": battery_tier,
        "battery_pct_actual": battery_pct_actual,
        "battery_saver_active": battery_saver_active,
        "thermal_condition": thermal_condition,
        "thermal_zone_used": device.thermal_zone_used(),
        "cpu_temp_start_c": round(temp_start, 2),
        "cpu_temp_end_c": round(temp_end, 2),
        "throttled": throttled,
        "freq_drop_pct": freq_drop_after,
        "prefill_tokens_per_sec": bench.prefill_tokens_per_sec,
        "decode_tokens_per_sec": bench.decode_tokens_per_sec,
        "prompt_tokens": bench.prompt_tokens,
        "generated_tokens": bench.generated_tokens,
        "model_load_ms": bench.model_load_ms,
        "prefill_duration_ms": bench.prefill_duration_ms,
        "decode_duration_ms": bench.decode_duration_ms,
        "watts_mean": watts_stats.mean,
        "watts_min": watts_stats.min,
        "watts_max": watts_stats.max,
        "watts_n_samples": watts_stats.n_samples,
        "watts_samples_json": json.dumps(watts_stats.raw_samples),
        "baseline_watts_mean": baseline_stats.mean,
        "watts_delta_mean": watts_delta_mean,
        "cache_state": cache_state,
        "swap_latency_ms": None,  # not applicable as-is with this invocation; revisit if config swap is measured separately
        "is_warmup": is_warmup,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default=None, help="adb serial/address if multiple devices are connected")
    parser.add_argument("--dry-run", action="store_true", help="Simulates adb with no connected device (script testing)")
    parser.add_argument("--reps", type=int, default=REPS_PER_CONFIG, help="Repetitions per configuration")
    parser.add_argument("--quantizations", nargs="+", default=QUANTIZATIONS, choices=list(QUANT_TO_PTE.keys()))
    parser.add_argument("--threads", nargs="+", type=int, default=THREADS)
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed to randomize run order")
    parser.add_argument("--probe-device", action="store_true",
                         help="Only prints device info (thermal zones, battery) and exits")
    parser.add_argument("--check-throttle-access", action="store_true",
                         help="Checks whether reading scaling_cur_freq/cpuinfo_max_freq is accessible "
                              "(not blocked by SELinux like thermal_zone) and exits. Run this before "
                              "any real collection to know whether throttled will be reliable or approximate.")
    parser.add_argument("--idle-baseline-s", type=float, default=2.0,
                         help="Duration (s) of the resting power measurement (screen off) right "
                              "before each run, to compute watts_delta_mean = watts_mean - baseline. "
                              "0 disables it (faster runs but loses the point of comparison).")
    parser.add_argument("--inter-rep-pause-s", type=float, default=0.0,
                         help="Pause (s) before each real repetition (excluding warmup), to induce a "
                              "'cold' cache state instead of back-to-back warm runs. 0 = historical "
                              "back-to-back behavior (cache_state='warm'). "
                              ">0 tags runs as cache_state='cold'.")
    args = parser.parse_args()

    device = Device(serial=args.serial, dry_run=args.dry_run)

    if args.probe_device:
        print("Battery:", device.battery_pct(), "%")
        print("CPU thermal zone used:", device.thermal_zone_used())
        print("CPU temperature:", device.cpu_temp_c(), "°C")
        print("Estimated watts:", device.battery_watts())
        print("\nManually verify these values (dumpsys battery, cat on the listed "
              "sysfs files) before running the full collection.")
        return

    if args.check_throttle_access:
        pairs = device._cpu_freqs()
        if pairs:
            print(f"[ok] cpufreq reading available on {len(pairs)} core(s).")
            print(f"     Example (cur, max) per core: {pairs[:4]}")
            busiest_cur, busiest_max = max(pairs, key=lambda p: p[0])
            print(f"     Most active core: cur={busiest_cur}, max={busiest_max} "
                  f"(drop={round(100.0 * (1 - busiest_cur / busiest_max), 2) if busiest_max else 0.0}%)")
            print("\nThrottling detected during the collection will be based on the real CPU "
                  "frequency, paired by core (fixed Day 3 — see EXECUTION_LOG.md).")
        else:
            print("[WARNING] cpufreq reading empty or refused by the system (probably "
                  "SELinux, like thermal_zone on the Pixel 7a — see README-collecte.md).")
            print("throttled will be estimated SOLELY from the temperature drop observed during the "
                  "run, not the real CPU frequency — less reliable.")
            print("Document this explicitly in the write-up if this limitation persists before the "
                  "real collection, to avoid presenting throttled as more precise than it is.")
        return

    configs = list(itertools.product(args.quantizations, args.threads))
    completed = load_completed_keys(RESULTS_CSV)
    total_runs = len(configs) * len(BATTERY_TIERS) * len(THERMAL_CONDITIONS) * args.reps
    print(f"Matrix: {len(configs)} configs × {len(BATTERY_TIERS)} battery tiers × "
          f"{len(THERMAL_CONDITIONS)} thermal conditions × {args.reps} reps "
          f"= {total_runs} runs (+ warm-ups).")
    if args.idle_baseline_s > 0:
        print(f"Idle baseline enabled: +{args.idle_baseline_s}s per run (screen off before each "
              f"inference) — adds roughly {total_runs * args.idle_baseline_s / 60:.1f} min total.")
    cache_state = "cold" if args.inter_rep_pause_s > 0 else "warm"

    for tier_id, tier_label, check_fn in BATTERY_TIERS:
        battery_pct_actual = wait_for_battery_condition(device, tier_id, tier_label, check_fn)
        for condition_id, condition_label in THERMAL_CONDITIONS:
            wait_for_thermal_condition(device, condition_id, condition_label)

            block_configs = configs.copy()
            random.Random(args.shuffle_seed).shuffle(block_configs)  # avoids order bias/thermal drift

            for quant, threads in block_configs:
                for w in range(WARMUP_RUNS):
                    print(f"[warmup] {quant} / {threads} threads / {tier_id} / {condition_id}")
                    try:
                        row = run_single(device, quant, threads, tier_id, battery_pct_actual,
                                          condition_id, is_warmup=True, rep_idx=w,
                                          idle_baseline_s=args.idle_baseline_s, cache_state=cache_state)
                        append_row(RESULTS_CSV, row)
                    except AdbError as e:
                        print(f"[warmup error] {e}", file=sys.stderr)

                for rep in range(args.reps):
                    key = (quant, str(threads), tier_id, condition_id, str(rep))
                    if key in completed:
                        print(f"[skip - already done] {quant}/{threads}/{tier_id}/{condition_id}/rep{rep}")
                        continue
                    if args.inter_rep_pause_s > 0:
                        print(f"[pause] {args.inter_rep_pause_s}s to induce a cold cache before this rep.")
                        time.sleep(args.inter_rep_pause_s)
                    print(f"[run] {quant} / {threads} threads / {tier_id} / {condition_id} / rep {rep} "
                          f"/ cache={cache_state}")
                    try:
                        row = run_single(device, quant, threads, tier_id, battery_pct_actual,
                                          condition_id, is_warmup=False, rep_idx=rep,
                                          idle_baseline_s=args.idle_baseline_s, cache_state=cache_state)
                        append_row(RESULTS_CSV, row)
                        completed.add(key)
                    except AdbError as e:
                        print(f"[error] {quant}/{threads}: {e}", file=sys.stderr)

    print(f"\nCollection complete. Results in: {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()
