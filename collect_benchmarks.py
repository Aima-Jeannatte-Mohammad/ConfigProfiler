#!/usr/bin/env python3
"""
collect_benchmarks.py — ConfigProfiler, étape 1 : collecte de données réelle.

ADAPTÉ POUR QWEN3 le 3 août 2026 (Jour 2). Voir EXECUTION_LOG.md pour le détail
de ce qui a changé par rapport à la version générique initiale.

CE QUI EST RÉUTILISÉ SANS MODIFICATION (validé sur Pixel 7a réel, Jour 1) :
    - classe Device entière : battery_pct, battery_watts, cpu_temp_c (via
      dumpsys thermalservice, contournement SELinux documenté),
      max_freq_drop_pct. Ne pas retoucher sans re-justifier par écrit.

CE QUI A ÉTÉ RÉÉCRIT (spécifique à Qwen3, absent de la version générique) :
    - run_on_device_benchmark() : appelle réellement `llama_main` (runner
      ARM64 cross-compilé, validé Jour 2 — voir EXECUTION_LOG.md, résultats
      73.45/22.00 tok/s en USB, 103.17/36.81 tok/s en Wi-Fi/discharging),
      avec le chat template Qwen3 (<|im_start|>...) et parsing du JSON
      PyTorchObserver réellement émis par le runner (plus fiable que l'ancien
      regex générique tokens_per_sec:, qui ne correspondait à aucune sortie
      réelle du runner ExecuTorch).
    - Mesure des watts : CORRIGÉE le Jour 2 après détection d'un défaut
      méthodologique sur le premier run réel (watts anormalement bas pendant
      l'inférence vs. à l'arrêt — voir EXECUTION_LOG.md). L'ancienne version
      appelait battery_watts() une seule fois, APRÈS la fin de l'inférence, ce
      qui mesurait un état déjà redescendu, pas la consommation réelle pendant
      le run. Corrigé par échantillonnage en parallèle (thread dédié, ~toutes
      les 0.4s) PENDANT toute la durée de l'appel bloquant à llama_main. Le
      CSV rapporte désormais watts_mean/watts_min/watts_max/watts_n_samples,
      pas une seule valeur post-hoc.

APPROFONDISSEMENTS MÉTHODOLOGIQUES ajoutés Jour 2 (au-delà de la correction
watts ci-dessus), en réponse à "comment aller plus loin pour un jury exigeant" :
    1. Baseline idle (--idle-baseline-s, défaut 2s) : mesure la consommation
       au repos (écran éteint, aucune inférence) juste avant chaque run.
       watts_delta_mean = watts_mean - baseline_watts_mean donne le coût
       attribuable à l'inférence elle-même, pas une valeur brute sans
       référence.
    2. Contrôle de l'état de l'écran (Device.ensure_screen_off) : variable
       confondante non maîtrisée jusqu'ici — un petit modèle quantizé peut
       consommer moins que l'écran allumé. Forcé éteint avant chaque baseline.
    3. Dimension cache_state (cold/warm) via --inter-rep-pause-s : l'écart de
       tok/s observé entre runs isolés (Jour 1-2, 22-37 tok/s) et runs
       enchaînés (Jour 2, ~48 tok/s) n'est plus une anomalie non expliquée
       mais une variable expérimentale explicite. 0 = comportement historique
       (runs enchaînés, cache_state="warm"), >0 = pause avant chaque rep pour
       induire un état "cold".
    4. --check-throttle-access : diagnostic à lancer AVANT toute vraie
       collecte pour savoir si la lecture scaling_cur_freq/cpuinfo_max_freq
       est bloquée par SELinux (comme thermal_zone) — évite de présenter
       throttled comme plus fiable qu'il ne l'est si cette lecture échoue
       silencieusement en repli sur 0.0.

⚠️ TODO AVANT LA VRAIE COLLECTE — un seul point restant :
    1. Mapping quantization -> fichier .pte : à ce jour, un seul .pte existe
       sur le device (qwen3_0_6b.pte, config qwen3_xnnpack_q8da4w). Pour une
       vraie matrice int8/int4, il faut exporter et pousser un .pte distinct
       par niveau de quantization avant de lancer la collecte complète.

DÉCISION TRANCHÉE (n'est plus un TODO) : mode "thinking" de Qwen3 désactivé
via le soft switch "/no_think" ajouté à chaque prompt utilisateur. Raison :
pollue la démo vidéo (séquence C) et fausse les métriques de tokens/latence
du dataset de fidélité jargon sans apporter de valeur au projet (on mesure la
fidélité de traduction, pas la qualité du raisonnement interne). Voir
EXECUTION_LOG.md pour la discussion complète.

Aucune dépendance externe : Python 3.8+, adb dans le PATH, USB ou débogage
sans fil activé.
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
# Matrice de configuration
# --------------------------------------------------------------------------
#
# Réduite volontairement par rapport à la version générique initiale
# ("profondeur plutôt que largeur", décision prise en amont pour tenir le
# calendrier solo à 13 jours — voir échange de planification, Jour 1).
# "mixed" retiré : pas de valeur ajoutée suffisante pour le coût solo.
# Threads limités à 2/4 : voir TODO #1 ci-dessus avant d'activer réellement.

QUANTIZATIONS = ["q8da4w"]   # TODO #2 : ajouter "int4" une fois ce .pte exporté et poussé
THREADS = [2, 4]

# Chemin du .pte sur le DEVICE (pas en local) pour chaque quantization.
# À compléter au fur et à mesure des exports (voir TODO #2).
QUANT_TO_PTE = {
    "q8da4w": "qwen3_0_6b.pte",
}

# (id_technique, description_affichée_à_l_utilisateur, contrainte_de_vérification)
BATTERY_TIERS = [
    ("high", ">70%", lambda pct: pct > 70),
    ("mid", "30-50%", lambda pct: 30 <= pct <= 50),
    ("low", "<20%", lambda pct: pct < 20),
]

THERMAL_CONDITIONS = [
    ("ambient", "température ambiante normale, pas de préchauffe"),
    ("preheated", "téléphone préchauffé 15-20 min (soleil/source de chaleur), "
                  "démarrage du run à >=40°C sur la thermal_zone CPU"),
]

REPS_PER_CONFIG = 3          # répétitions par (quant, threads, batterie, thermique)
WARMUP_RUNS = 1              # runs jetés avant les vraies mesures, par bloc
PREHEAT_MIN_TEMP_C = 40.0

# Dossier sur le DEVICE où modèle/tokenizer/runner sont déjà déployés (Jour 1-2)
DEVICE_DIR = "/data/local/tmp/configprofiler"
TOKENIZER_FILENAME = "tokenizer.json"
RUNNER_FILENAME = "llama_main"

# Prompt de test — un seul pour l'instant (mesure de perf pure, pas de
# fidélité jargon ici, c'est un dataset séparé, section 5 du document de
# référence, pas encore attaqué)
DEFAULT_PROMPT = "Who is the president of the US?"

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "configprofiler_dataset.csv"

CSV_FIELDS = [
    "run_id", "timestamp_iso", "quantization", "threads",
    "battery_tier", "battery_pct_actual", "thermal_condition",
    "thermal_zone_used", "cpu_temp_start_c", "cpu_temp_end_c",
    "throttled", "freq_drop_pct", "prefill_tokens_per_sec",
    "decode_tokens_per_sec", "prompt_tokens", "generated_tokens",
    "model_load_ms", "prefill_duration_ms", "decode_duration_ms",
    "watts_mean", "watts_min", "watts_max",
    "watts_n_samples", "watts_samples_json", "baseline_watts_mean",
    "watts_delta_mean", "cache_state", "swap_latency_ms", "is_warmup", "notes",
]

# --------------------------------------------------------------------------
# Couche ADB / device — INCHANGÉE depuis la version générique, validée sur
# Pixel 7a réel Jour 1 (corrections SELinux/thermal et biais USB/watts déjà
# documentées et testées, voir README-collecte.md). Ne pas modifier sans
# revalider sur device réel et documenter le changement dans EXECUTION_LOG.md.
# --------------------------------------------------------------------------

class AdbError(RuntimeError):
    pass


class Device:
    """Wrapper fin autour d'adb shell. Pas de dépendance externe."""

    def __init__(self, serial: Optional[str] = None, dry_run: bool = False):
        self.serial = serial
        self.dry_run = dry_run
        self._thermal_zone_cache: Optional[str] = None
        self._current_sign_cache: Optional[int] = None  # +1 ou -1
        self._mock_temp_c: float = 36.0  # monte à chaque lecture en dry-run pour simuler une préchauffe
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
            raise AdbError(f"Timeout sur: {command}") from e
        if result.returncode != 0 and not result.stdout:
            raise AdbError(f"adb shell a échoué ({command}): {result.stderr.strip()}")
        return result.stdout

    def _mock_shell(self, command: str) -> str:
        """Réponses simulées pour tester le script sans appareil connecté (--dry-run)."""
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
            # Commande combinée (Jour 2, optimisation échantillonnage) :
            # `cat f1 f2` concatène les deux fichiers, une valeur par ligne.
            return "-850000\n3900000\n"
        if "current_now" in command:
            return "-850000\n"
        if "voltage_now" in command:
            return "3900000\n"
        if "scaling_cur_freq" in command:
            return "1800000\n"
        if "cpuinfo_max_freq" in command:
            return "2400000\n"
        if "dumpsys power" in command:
            return "  mWakefulness=Awake\n"
        if "input keyevent" in command:
            return ""
        if RUNNER_FILENAME in command:
            # simule une ligne PyTorchObserver plausible
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

    # ---- Batterie ----

    def battery_pct(self) -> int:
        out = self.shell("dumpsys battery")
        m = re.search(r"level:\s*(\d+)", out)
        if not m:
            raise AdbError("Impossible de lire le niveau de batterie via dumpsys battery")
        return int(m.group(1))

    def battery_watts(self) -> float:
        """Estimation en Watts à partir de current_now (µA) et voltage_now (µV).

        ATTENTION : le signe de current_now dépend du firmware (positif ou négatif
        en décharge selon les OEM). On calibre le signe au premier appel et on le
        fige pour le reste de la session — vérifiez la cohérence avec un test
        manuel (débrancher le chargeur, vérifier que la valeur affichée a du sens).

        OPTIMISATION Jour 2 : les deux fichiers sont lus en un seul appel adb
        (`cat f1 f2`) plutôt que deux appels séquentiels. Découverte en creusant
        la série temporelle watts_samples_json : deux appels adb par échantillon
        faisaient que l'intervalle réel entre échantillons (~1.0-1.4s) dépassait
        largement le paramètre interval_s=0.4 du WattsSampler — voir
        EXECUTION_LOG.md. Un seul appel adb double approximativement la
        résolution temporelle obtenue.
        """
        out = self.shell(
            "cat /sys/class/power_supply/battery/current_now "
            "/sys/class/power_supply/battery/voltage_now"
        ).strip()
        lines = [l for l in out.splitlines() if l.strip()]
        if len(lines) < 2:
            raise AdbError(
                f"Sortie inattendue pour current_now/voltage_now combinés "
                f"(attendu 2 lignes, reçu {len(lines)}): {out!r}"
            )
        cur_raw, volt_raw = lines[0].strip(), lines[1].strip()
        current_ua = float(cur_raw)
        voltage_uv = float(volt_raw)
        if self._current_sign_cache is None:
            self._current_sign_cache = -1 if current_ua < 0 else 1
        current_a = abs(current_ua) / 1e6
        voltage_v = abs(voltage_uv) / 1e6
        return round(current_a * voltage_v, 4)

    # ---- Thermique ----
    #
    # NOTE IMPORTANTE (découverte en collecte réelle sur Pixel 7a, pas une
    # hypothèse) : l'accès direct à /sys/class/thermal/thermal_zone*/temp est
    # bloqué par SELinux pour l'utilisateur "shell" (celui utilisé par adb),
    # même sans root — comportement documenté sur les Pixel récents, pas un
    # bug de ce script. La source utilisée à la place est `dumpsys
    # thermalservice`, un service Android accessible sans root via adb shell.
    #
    # Sur le Tensor G2 (Pixel 7a), ce service expose les trois clusters CPU
    # sous les noms LITTLE / MID / BIG (mType=0 = CPU dans l'énumération
    # Android ThermalHAL), dans la section "Current temperatures from HAL"
    # (PAS "Cached temperatures", qui contient des valeurs obsolètes/stables
    # non représentatives de l'état courant — vérifié sur sortie réelle).

    _THERMAL_HAL_CPU_TYPE = "0"  # mType=0 == CPU dans l'énumération Android ThermalHAL

    def _parse_thermal_hal_cpu(self) -> dict[str, float]:
        """Extrait les températures des clusters CPU depuis la section
        'Current temperatures from HAL' de dumpsys thermalservice."""
        out = self.shell("dumpsys thermalservice")
        if "Current temperatures from HAL" not in out:
            raise AdbError(
                "Section 'Current temperatures from HAL' absente de dumpsys "
                "thermalservice. Sortie complète à inspecter manuellement."
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
                "Aucune entrée mType=0 (CPU) trouvée dans 'Current temperatures "
                "from HAL'. Inspectez manuellement 'adb shell dumpsys "
                "thermalservice' et adaptez _THERMAL_HAL_CPU_TYPE / le parsing "
                "si votre device nomme ses clusters différemment."
            )
        return cpu_temps

    def cpu_temp_c(self) -> float:
        """Retourne la température du cluster CPU le plus chaud (le plus
        pertinent pour détecter un throttling imminent)."""
        cpu_temps = self._parse_thermal_hal_cpu()
        if self._thermal_zone_cache is None:
            self._thermal_zone_cache = ", ".join(sorted(cpu_temps.keys()))
            print(f"[device] Clusters CPU détectés via dumpsys thermalservice: "
                  f"{self._thermal_zone_cache}", file=sys.stderr)
        return max(cpu_temps.values())

    def thermal_zone_used(self) -> str:
        if self._thermal_zone_cache is None:
            cpu_temps = self._parse_thermal_hal_cpu()
            self._thermal_zone_cache = ", ".join(sorted(cpu_temps.keys()))
        return f"dumpsys thermalservice (HAL, mType=0): {self._thermal_zone_cache}"

    # ---- CPU freq / throttling ----

    def _cpu_freqs(self) -> tuple[list[int], list[int]]:
        """Retourne (freqs_courantes, freqs_max) pour tous les cpuN présents.

        NOTE : sur ce projet, l'accès direct à /sys/class/thermal a été trouvé
        bloqué par SELinux pour l'utilisateur shell sur Pixel 7a (voir section
        Thermique ci-dessus) — il est possible que /sys/devices/system/cpu/...
        le soit aussi selon la version d'Android. Si c'est le cas, cette
        méthode retourne des listes vides plutôt que de faire planter toute la
        collecte ; freq_drop_pct sera alors 0.0 pour tous les runs et
        throttled sera basé uniquement sur la chute observée pendant le run
        (moins fiable) — à vérifier manuellement avec la commande ci-dessous
        avant de lancer une vraie session de collecte.
        """
        if self._cpufreq_unavailable:
            return [], []
        try:
            cur_out = self.shell(
                "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; "
                "do cat $f 2>/dev/null; done"
            )
            max_out = self.shell(
                "for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; "
                "do cat $f 2>/dev/null; done"
            )
        except AdbError:
            cur_out, max_out = "", ""
        cur = [int(x) for x in cur_out.split() if x.strip().isdigit()]
        mx = [int(x) for x in max_out.split() if x.strip().isdigit()]
        if not cur or not mx:
            self._cpufreq_unavailable = True
            print("[device] AVERTISSEMENT: lecture de scaling_cur_freq/cpuinfo_max_freq "
                  "vide ou refusée. Vérifiez manuellement avec: adb shell "
                  "'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; "
                  "do echo $f:$(cat $f); done' — si bloqué par SELinux comme pour "
                  "thermal_zone, throttled sera estimé uniquement via la chute de "
                  "température pendant le run, pas via la fréquence CPU.",
                  file=sys.stderr)
        return cur, mx

    def max_freq_drop_pct(self) -> float:
        """% de chute entre la fréquence max théorique et la fréquence courante
        du cœur le plus rapide au moment de l'appel. Positif = ralentissement."""
        cur, mx = self._cpu_freqs()
        if not cur or not mx:
            return 0.0
        best_cur = max(cur)
        best_max = max(mx)
        if best_max == 0:
            return 0.0
        return round(100.0 * (1 - best_cur / best_max), 2)

    # ---- Écran — ajouté Jour 2 : variable confondante non contrôlée
    # jusqu'ici. Un petit modèle quantizé peut consommer moins que l'écran
    # allumé, donc un état d'écran non maîtrisé entre runs invaliderait
    # silencieusement les comparaisons de watts.

    def is_screen_on(self) -> bool:
        out = self.shell("dumpsys power")
        return "mWakefulness=Awake" in out

    def ensure_screen_off(self) -> None:
        """Force l'écran éteint s'il est allumé. Idempotent — ne fait rien
        si déjà éteint (évite de le rallumer par erreur avec un toggle
        aveugle)."""
        if self.is_screen_on():
            self.shell("input keyevent 26")  # touche power, toggle
            time.sleep(0.5)


# --------------------------------------------------------------------------
# Échantillonnage des watts PENDANT l'inférence — correction méthodologique
# du Jour 2 (voir docstring en tête de fichier et EXECUTION_LOG.md).
#
# L'ancienne approche appelait device.battery_watts() une seule fois, après
# la fin du run — ce qui mesure un état déjà redescendu, pas la consommation
# réelle pendant l'inférence. Ce sampler tourne dans un thread séparé pendant
# toute la durée de l'appel bloquant à llama_main, via des appels adb
# indépendants (adb multiplexe plusieurs sessions shell sans conflit).
# --------------------------------------------------------------------------

class WattsSampler:
    """Échantillonne device.battery_watts() en continu dans un thread, tant
    que .stop() n'a pas été appelé. Conçu pour encadrer un appel bloquant
    (ex. l'inférence llama_main) et mesurer la consommation PENDANT le run,
    pas seulement avant/après.

    Chaque échantillon est horodaté en secondes relatives depuis .start()
    (horloge HÔTE/WSL, pas celle du device) — permet de reconstruire une
    courbe puissance/temps par run pour analyse ultérieure. ⚠️ Cet
    horodatage n'est PAS synchronisé avec les timestamps du JSON
    PyTorchObserver (horloge du DEVICE) — voir limite documentée dans
    EXECUTION_LOG.md avant toute tentative de corréler un échantillon watts
    à une phase precise (prefill/decode) du run.
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
                # Une lecture ratée pendant l'échantillonnage ne doit pas
                # faire planter tout le run — on la saute simplement.
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
            # Filet de sécurité : run trop court pour un seul échantillon
            # (peu probable avec interval_s=0.4s et une inférence de
            # plusieurs secondes, mais on ne veut pas planter dessus).
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
    """Mesure la consommation au repos (écran éteint, aucune inférence) juste
    avant un run, pour calculer un delta watts_inference - watts_idle plutôt
    que de présenter une valeur brute sans point de comparaison. Ajouté Jour 2
    suite à la remarque sur l'approfondissement méthodologique (voir
    EXECUTION_LOG.md).

    duration_s=0 désactive la mesure (retourne des NaN) — utile pour garder
    les runs rapides pendant le développement/debug du script.
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
    """Construit la commande shell exécutée sur le device.

    Flag confirmé via `llama_main --help` sur le device (Jour 2) :
    -cpu_threads (int32, défaut -1 = heuristique automatique). gflags accepte
    aussi bien -cpu_threads=N que --cpu_threads=N (testé avec --tokenizer_path
    dans les runs précédents) — on garde le style double-tiret pour la
    cohérence avec le reste du projet.

    DÉCISION Jour 2 (tranchée, voir EXECUTION_LOG.md) : mode "thinking" de
    Qwen3 désactivé via le soft switch "/no_think" ajouté au message
    utilisateur. Le hard switch (enable_thinking=False) est une option
    Python côté tokenizer HuggingFace (apply_chat_template), inutilisable
    depuis ce runner C++ qui construit le prompt en texte brut — le soft
    switch textuel est donc la seule option disponible ici. Confirmé
    fonctionnel pour Qwen3 (pas Qwen3-VL, pas Qwen3.5, qui ont un
    comportement différent).
    """
    if quantization not in QUANT_TO_PTE:
        raise ValueError(
            f"Quantization '{quantization}' non mappée à un .pte connu. "
            f"Mappings disponibles: {list(QUANT_TO_PTE.keys())}. "
            f"Exporter et pousser le .pte manquant avant de continuer (TODO #2)."
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
    """Lance llama_main sur le device et parse la ligne JSON PyTorchObserver
    réellement émise par le runner (confirmé Jour 2, voir EXECUTION_LOG.md).
    """
    cmd = build_llama_main_command(quantization, threads, prompt)
    out = device.shell(cmd, timeout=120.0)

    json_match = re.search(r"PyTorchObserver\s+(\{.*\})", out)
    if not json_match:
        raise AdbError(
            f"Ligne PyTorchObserver introuvable dans la sortie de llama_main. "
            f"Sortie brute (dernières 500 car.):\n{out[-500:]}"
        )
    try:
        metrics = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        raise AdbError(f"JSON PyTorchObserver malformé: {e}\nLigne: {json_match.group(1)}") from e

    model_load_ms = metrics.get("model_load_end_ms", 0) - metrics.get("model_load_start_ms", 0)

    # Durées prefill/decode réelles, calculées uniquement à partir de
    # timestamps DEVICE (même horloge des deux côtés du calcul, donc fiable
    # — contrairement à toute tentative de corréler ça avec les échantillons
    # watts côté hôte, dont l'horloge n'est pas synchronisée avec celle du
    # device. Voir EXECUTION_LOG.md pour la limite documentée sur ce point.
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
# Orchestration — INCHANGÉE (logique de paliers/warmup/reprise déjà solide)
# --------------------------------------------------------------------------

def load_completed_keys(csv_path: Path) -> set[tuple]:
    """Permet de reprendre une collecte interrompue sans dupliquer des runs."""
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
            print(f"[ok] Batterie à {pct}% — conforme au palier '{tier_id}' ({tier_label}).")
            return pct
        print(f"[attente] Batterie actuelle: {pct}% — besoin de {tier_label} pour le palier "
              f"'{tier_id}'. Ajustez la charge (branchez/débranchez) et appuyez sur Entrée "
              f"pour re-vérifier (ou tapez 'force' pour continuer quand même).")
        answer = input("> ").strip().lower()
        if answer == "force":
            print(f"[forcé] Poursuite avec batterie à {pct}%, hors du palier théorique "
                  f"— sera noté dans les notes du run.")
            return pct


def wait_for_thermal_condition(device: Device, condition_id: str, condition_label: str) -> None:
    if condition_id != "preheated":
        input(f"[setup] Condition thermique '{condition_id}' ({condition_label}). "
              f"Assurez un état ambiant stable, puis Entrée pour continuer.")
        return
    while True:
        temp = device.cpu_temp_c()
        if temp >= PREHEAT_MIN_TEMP_C:
            print(f"[ok] Température CPU de départ: {temp:.1f}°C (seuil: {PREHEAT_MIN_TEMP_C}°C).")
            return
        print(f"[attente] Température CPU actuelle: {temp:.1f}°C, seuil requis: "
              f"{PREHEAT_MIN_TEMP_C}°C. Poursuivez la préchauffe puis Entrée pour re-vérifier.")
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

    # Baseline idle (écran éteint, aucune inférence) — ajouté Jour 2. Mesuré
    # juste avant l'inférence pour donner un point de comparaison réel au
    # watts_mean pendant le run (delta = coût attribuable à l'inférence,
    # pas une valeur brute sans référence). idle_baseline_s=0 désactive.
    baseline_stats = measure_idle_baseline(device, duration_s=idle_baseline_s)

    # Échantillonnage watts PENDANT l'inférence (correction Jour 2) : le
    # sampler tourne dans un thread séparé, encadrant précisément l'appel
    # bloquant à llama_main — pas de mesure isolée avant/après seulement.
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
        notes += "; ATTENTION: aucun échantillon watts collecté pendant ce run (voir watts_n_samples)"
    if idle_baseline_s <= 0:
        notes += "; baseline idle désactivée (idle_baseline_s=0), watts_delta_mean non calculable"

    return {
        "run_id": run_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "quantization": quantization,
        "threads": threads,
        "battery_tier": battery_tier,
        "battery_pct_actual": battery_pct_actual,
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
        "swap_latency_ms": None,  # non applicable tel quel avec cette invocation ; à revoir si le swap de config est mesuré séparément
        "is_warmup": is_warmup,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default=None, help="Numéro de série/adresse adb si plusieurs appareils connectés")
    parser.add_argument("--dry-run", action="store_true", help="Simule adb sans appareil connecté (test du script)")
    parser.add_argument("--reps", type=int, default=REPS_PER_CONFIG, help="Répétitions par configuration")
    parser.add_argument("--quantizations", nargs="+", default=QUANTIZATIONS, choices=list(QUANT_TO_PTE.keys()))
    parser.add_argument("--threads", nargs="+", type=int, default=THREADS)
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Graine pour randomiser l'ordre des runs")
    parser.add_argument("--probe-device", action="store_true",
                         help="Affiche uniquement les infos device (thermal zones, batterie) et quitte")
    parser.add_argument("--check-throttle-access", action="store_true",
                         help="Vérifie si la lecture scaling_cur_freq/cpuinfo_max_freq est accessible "
                              "(pas bloquée par SELinux comme thermal_zone) et quitte. À lancer avant "
                              "toute vraie collecte pour savoir si throttled est fiable ou approximatif.")
    parser.add_argument("--idle-baseline-s", type=float, default=2.0,
                         help="Durée (s) de mesure de la consommation au repos (écran éteint) juste "
                              "avant chaque run, pour calculer watts_delta_mean = watts_mean - baseline. "
                              "0 désactive (accélère les runs mais perd le point de comparaison).")
    parser.add_argument("--inter-rep-pause-s", type=float, default=0.0,
                         help="Pause (s) avant chaque répétition réelle (hors warmup), pour induire un "
                              "état de cache 'froid' plutôt que des runs enchaînés à chaud. 0 = runs "
                              "enchaînés normalement (cache_state='warm', comportement historique). "
                              ">0 tague les runs cache_state='cold'.")
    args = parser.parse_args()

    device = Device(serial=args.serial, dry_run=args.dry_run)

    if args.probe_device:
        print("Batterie:", device.battery_pct(), "%")
        print("Thermal zone CPU utilisée:", device.thermal_zone_used())
        print("Température CPU:", device.cpu_temp_c(), "°C")
        print("Watts estimés:", device.battery_watts())
        print("\nVérifiez ces valeurs manuellement (dumpsys battery, cat sur les fichiers "
              "sysfs listés) avant de lancer la collecte complète.")
        return

    if args.check_throttle_access:
        cur, mx = device._cpu_freqs()
        if cur and mx:
            print(f"[ok] Lecture cpufreq disponible sur {len(cur)} cœur(s).")
            print(f"     Exemple fréquences courantes: {cur[:4]}")
            print(f"     Exemple fréquences max:       {mx[:4]}")
            print("\nLe throttling détecté pendant la collecte sera basé sur la vraie fréquence CPU.")
        else:
            print("[AVERTISSEMENT] Lecture cpufreq vide ou refusée par le système (probablement "
                  "SELinux, comme pour thermal_zone sur Pixel 7a — voir README-collecte.md).")
            print("throttled sera estimé UNIQUEMENT via la chute de température observée pendant le "
                  "run, pas via la fréquence CPU réelle — moins fiable.")
            print("À documenter explicitement dans le write-up si cette limite persiste avant la "
                  "vraie collecte, pour ne pas présenter throttled comme plus précis qu'il ne l'est.")
        return

    configs = list(itertools.product(args.quantizations, args.threads))
    completed = load_completed_keys(RESULTS_CSV)
    total_runs = len(configs) * len(BATTERY_TIERS) * len(THERMAL_CONDITIONS) * args.reps
    print(f"Matrice: {len(configs)} configs × {len(BATTERY_TIERS)} paliers batterie × "
          f"{len(THERMAL_CONDITIONS)} conditions thermiques × {args.reps} reps "
          f"= {total_runs} runs (+ warm-ups).")
    if args.idle_baseline_s > 0:
        print(f"Baseline idle activée: +{args.idle_baseline_s}s par run (écran éteint avant chaque "
              f"inférence) — ajoute environ {total_runs * args.idle_baseline_s / 60:.1f} min au total.")
    cache_state = "cold" if args.inter_rep_pause_s > 0 else "warm"

    for tier_id, tier_label, check_fn in BATTERY_TIERS:
        battery_pct_actual = wait_for_battery_condition(device, tier_id, tier_label, check_fn)
        for condition_id, condition_label in THERMAL_CONDITIONS:
            wait_for_thermal_condition(device, condition_id, condition_label)

            block_configs = configs.copy()
            random.Random(args.shuffle_seed).shuffle(block_configs)  # évite le biais d'ordre/dérive thermique

            for quant, threads in block_configs:
                for w in range(WARMUP_RUNS):
                    print(f"[warmup] {quant} / {threads} threads / {tier_id} / {condition_id}")
                    try:
                        row = run_single(device, quant, threads, tier_id, battery_pct_actual,
                                          condition_id, is_warmup=True, rep_idx=w,
                                          idle_baseline_s=args.idle_baseline_s, cache_state=cache_state)
                        append_row(RESULTS_CSV, row)
                    except AdbError as e:
                        print(f"[erreur warmup] {e}", file=sys.stderr)

                for rep in range(args.reps):
                    key = (quant, str(threads), tier_id, condition_id, str(rep))
                    if key in completed:
                        print(f"[skip - déjà fait] {quant}/{threads}/{tier_id}/{condition_id}/rep{rep}")
                        continue
                    if args.inter_rep_pause_s > 0:
                        print(f"[pause] {args.inter_rep_pause_s}s pour induire un cache froid avant ce rep.")
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
                        print(f"[erreur] {quant}/{threads}: {e}", file=sys.stderr)

    print(f"\nCollecte terminée. Résultats dans: {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()
