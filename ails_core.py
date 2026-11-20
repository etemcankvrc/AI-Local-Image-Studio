# -*- coding: utf-8 -*-
"""
AI Local Image Studio — çekirdek modül (arayüzden bağımsız).

v1.0.1 ilkeleri:
  * Tek konum kuralı: programın indirdiği HER ŞEY (görsel motoru, önbellekler, geçici dosyalar,
    modeller, çıktılar) kullanıcının seçtiği veri klasörüne gider. Genel Python'a hiçbir şey kurulmaz;
    program kendi sanal ortamını (runtime\\venv) kullanır.
  * Hiçbir büyük indirme kullanıcı onayı olmadan başlamaz; indirmeden önce disk alanı kontrol edilir.
  * PyTorch ana süreçte ancak ayrı bir süreçte sağlam olduğu doğrulandıktan sonra içe aktarılır.
  * Ollama'nın kendi model konumu ayarına uyulur; ağdaki Ollama sunucuları ağı yormadan bulunur.

v1.0.2:
  * Daha yüksek adım ön ayarları + Ultra + elle adım, SD ailesinde DPM++ 2M Karras örnekleyici.
  * İkinci geçiş (hires fix) ve yüz/el düzeltici: ikisi de yüklü modelin bileşenlerini paylaşan img2img
    pipeline'ı kullanır (ek bellek yok). Algılama CPU'da OpenCV ile yapılır.
  * v2 hazırlığı: stil veri setleri, yerel içe aktarma, kibar (tek bağlantı, 1 sn aralıklı) URL indirme.

Bu modül içe aktarılırken yalnızca standart kütüphane kullanır; requests, torch, diffusers
gibi paketler ihtiyaç anında içe aktarılır.
"""
import os
import sys
import json
import re
import gc
import glob
import time
import math
import shutil
import socket
import threading
import subprocess
import importlib
import importlib.util
import ipaddress
import hashlib
import warnings

# Kütüphanelerin kullanıcıyı ilgilendirmeyen "ileride kaldırılacak" uyarıları log'u kirletmesin.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Already found a `peft_config` attribute.*")

# GPU sıralaması nvidia-smi ile aynı olsun (çoklu GPU'da doğru kartı seçmek için).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Hugging Face sunucusu hesapsız indirmelerde 'X-HF-Warning' başlığıyla "unauthenticated requests ... set a HF_TOKEN"
# notu gönderiyor; huggingface_hub bunu konsola uyarı olarak basıyor. Hata değil (indirme hesapsız da çalışır),
# kullanıcıyı korkutmasın diye gizlenir. Kullanıcı isterse kendi HF_TOKEN ortam değişkenini tanımlayabilir.
import logging as _logging


class _HfNoiseFilter(_logging.Filter):
    _PAT = re.compile(r"unauthenticated requests|set a HF_TOKEN|HF_TOKEN to enable", re.I)

    def filter(self, record):
        try:
            return not self._PAT.search(record.getMessage())
        except Exception:
            return True


HF_NOISE_FILTER = _HfNoiseFilter()
for _name in ("huggingface_hub.utils._http", "huggingface_hub.file_download", "huggingface_hub"):
    _logging.getLogger(_name).addFilter(HF_NOISE_FILTER)

# pythonw / pencereli exe'de stdout/stderr None olur; tqdm vb. çökmesin.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

IS_WIN = os.name == "nt"
GB = 1024 ** 3
MB = 1024 ** 2


def no_window_kwargs():
    """Windows'ta alt süreçlerin siyah konsol penceresi açmasını engeller."""
    if not IS_WIN:
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}


# =============================================================================
# Hata türleri
# =============================================================================
class AbortedError(Exception):
    """Kullanıcı durdurdu ya da Bellek Koruyucu işi güvenle kesti."""
    def __init__(self, reason="user"):
        super().__init__(reason)
        self.reason = reason


class MemoryGuardError(Exception):
    """İş başlamadan önce yetersiz bellek tespit edildi."""


class NeedRestartError(Exception):
    """Programın yeniden başlatılması gerekiyor."""


class DiskFullError(Exception):
    """Hedef diskte yeterli boş alan yok."""
    def __init__(self, drive, need_gb, free_gb):
        super().__init__(f"not enough space on {drive}: need {need_gb:.1f} GB, free {free_gb:.1f} GB")
        self.drive, self.need_gb, self.free_gb = drive, float(need_gb), float(free_gb)


class NetworkError(Exception):
    """İnternet / ağ bağlantısı sorunu."""


class EngineBrokenError(Exception):
    """Görsel motoru (PyTorch) bozuk ya da bu süreçte artık yüklenemiyor."""


ENGINE_BROKEN_HINTS = ("winerror 127", "winerror 126", "dll load failed", "c10_cuda.dll", "c10.dll",
                       "can only be initialized once", "error loading", "procedure could not be found",
                       "specified module could not be found")


def looks_like_engine_broken(exc):
    msg = str(exc).lower()
    return any(h in msg for h in ENGINE_BROKEN_HINTS)


def import_torch():
    """torch'u içe aktarır; bozuk kurulumu (DLL hatası vb.) EngineBrokenError olarak bildirir."""
    try:
        import torch  # noqa
        return torch
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa
        raise EngineBrokenError(f"{type(e).__name__}: {str(e)[:300]}") from e


# =============================================================================
# Veri konumu — programın indirdiği HER ŞEY kullanıcının seçtiği klasöre gider
# =============================================================================
DEFAULT_ROOT_NAME = "AI Local Image Studio"


class DataPaths:
    def __init__(self, root):
        j = os.path.join
        self.root = os.path.abspath(root)
        self.runtime = j(self.root, "runtime")
        self.venv = j(self.runtime, "venv")
        if IS_WIN:
            self.venv_python = j(self.venv, "Scripts", "python.exe")
            self.venv_pythonw = j(self.venv, "Scripts", "pythonw.exe")
        else:
            self.venv_python = j(self.venv, "bin", "python")
            self.venv_pythonw = self.venv_python
        self.engine_marker = j(self.runtime, "engine.json")
        self.models = j(self.root, "models")
        self.outputs = j(self.root, "outputs")
        self.cache = j(self.root, "cache")
        self.hf_home = j(self.cache, "huggingface")
        self.torch_home = j(self.cache, "torch")
        self.tmp = j(self.root, "tmp")
        self.logs = j(self.root, "logs")
        self.settings = j(self.root, "settings.json")
        self.ollama_dir = j(self.root, "ollama")
        self.ollama_models = j(self.root, "ollama_models")

    def ensure_dirs(self):
        for d in (self.runtime, self.models, self.outputs, self.cache, self.hf_home, self.tmp, self.logs):
            os.makedirs(d, exist_ok=True)
        try:
            os.makedirs(os.path.join(self.root, "addons", "models"), exist_ok=True)   # kullanıcının kendi modelleri
        except OSError:
            pass

    def env(self):
        """Program ve alt süreçleri (pip vb.) için ortam: önbellek ve geçici dosyalar da seçilen diske."""
        return {"HF_HOME": self.hf_home, "TORCH_HOME": self.torch_home,
                "TMP": self.tmp, "TEMP": self.tmp, "TMPDIR": self.tmp,
                "PIP_NO_CACHE_DIR": "1", "PYTHONNOUSERSITE": "1", "AILS_DATA_ROOT": self.root}

    def site_packages(self):
        if IS_WIN:
            return os.path.join(self.venv, "Lib", "site-packages")
        cands = sorted(glob.glob(os.path.join(self.venv, "lib", "python*", "site-packages")))
        return cands[-1] if cands else os.path.join(self.venv, "lib", "site-packages")


def apply_env(paths):
    paths.ensure_dirs()
    os.environ.update(paths.env())
    import tempfile
    tempfile.tempdir = paths.tmp


def sub_env(paths=None, drop_ours=False):
    env = os.environ.copy()
    if paths is not None:
        env.update(paths.env())
    if drop_ours:
        for k in ("PYTHONNOUSERSITE", "PIP_NO_CACHE_DIR", "AILS_DATA_ROOT"):
            env.pop(k, None)
    return env


# =============================================================================
# Disk
# =============================================================================
def _existing_parent(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def disk_free_gb(path):
    try:
        return shutil.disk_usage(_existing_parent(path)).free / GB
    except Exception:
        return -1.0


def drive_label(path):
    d = os.path.splitdrive(os.path.abspath(path))[0]
    return d if d else "/"


def same_drive(a, b):
    return os.path.normcase(drive_label(a)) == os.path.normcase(drive_label(b))


def check_space(path, need_gb):
    """Yer yoksa DiskFullError. Boş alan ölçülemiyorsa engellemez."""
    free = disk_free_gb(path)
    if 0 <= free < need_gb:
        raise DiskFullError(drive_label(path), need_gb, free)
    return free


def is_writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".ails_write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def list_drives():
    """[(kök, boş_GB, toplam_GB)] — Windows'ta sabit ve çıkarılabilir diskler."""
    out = []
    if IS_WIN:
        try:
            import ctypes
            mask = ctypes.windll.kernel32.GetLogicalDrives()
            for i in range(26):
                if not mask & (1 << i):
                    continue
                d = f"{chr(65 + i)}:\\"
                if ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(d)) not in (2, 3):
                    continue
                try:
                    u = shutil.disk_usage(d)
                except Exception:
                    continue
                out.append((d, u.free / GB, u.total / GB))
        except Exception:
            pass
    else:
        seen = set()
        for d in (os.path.expanduser("~"), "/"):
            try:
                st = os.stat(d)
                if st.st_dev in seen:
                    continue
                seen.add(st.st_dev)
                u = shutil.disk_usage(d)
                out.append((d, u.free / GB, u.total / GB))
            except Exception:
                pass
    return out


def suggest_root(drives=None):
    """En çok boş alanı olan diskte '<disk>\\AI Local Image Studio'."""
    if not IS_WIN:
        return os.path.join(os.path.expanduser("~"), DEFAULT_ROOT_NAME)
    drives = drives if drives is not None else list_drives()
    if not drives:
        return os.path.join(os.path.expanduser("~"), DEFAULT_ROOT_NAME)
    best = max(drives, key=lambda d: d[1])
    return os.path.join(best[0], DEFAULT_ROOT_NAME)


def folder_size_bytes(path, skip=()):
    total = 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    for root, dirs, files in os.walk(path):
        if skip and os.path.normpath(root) == os.path.normpath(path):
            dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clean_dir(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError:
            pass


# =============================================================================
# Donanım tespiti
# =============================================================================
class Hardware:
    def __init__(self):
        self.vendor = "none"        # "nvidia" | "none"
        self.gpu_name = ""
        self.gpu_index = 0
        self.vram_gb = 0.0
        self.compute_cap = 0.0
        self.driver = ""
        self.driver_major = 0
        self.ram_total_gb = 0.0
        self.pagefile_gb = -1.0     # -1 = bilinmiyor
        self.cuda_usable = False    # görsel motoru kurulu ve CUDA çalışıyor mu

    @property
    def device(self):
        return f"cuda:{self.gpu_index}" if self.cuda_usable else "cpu"


def _query_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return []
    for fields in (["name", "memory.total", "compute_cap", "driver_version"],
                   ["name", "memory.total", "driver_version"], ["name", "memory.total"]):
        try:
            out = subprocess.run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15, **no_window_kwargs())
            if out.returncode != 0:
                continue
            gpus = []
            for idx, line in enumerate(out.stdout.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < len(fields):
                    continue
                d = dict(zip(fields, parts))
                try:
                    mem_mb = float(d["memory.total"])
                except ValueError:
                    continue
                try:
                    cc = float(d.get("compute_cap", "0") or 0)
                except ValueError:
                    cc = 0.0
                gpus.append({"index": idx, "name": d["name"], "vram_gb": mem_mb / 1024.0, "cc": cc,
                             "driver": d.get("driver_version", "")})
            if gpus:
                return gpus
        except Exception:
            continue
    return []


def ram_total_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / GB
    except Exception:
        pass
    if IS_WIN:
        try:
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / GB
        except Exception:
            return 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / GB
    except Exception:
        pass
    return 0.0


def ram_available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / GB
    except Exception:
        return ram_total_gb() * 0.5


def pagefile_gb():
    try:
        import psutil
        return psutil.swap_memory().total / GB
    except Exception:
        return -1.0


def pagefile_free_gb():
    """Sayfa dosyasında (Windows) / takas alanında boş yer. Bilinmiyorsa -1."""
    try:
        import psutil
        sw = psutil.swap_memory()
        return sw.free / GB if sw.total > 0 else 0.0
    except Exception:
        return -1.0


def ram_reserve_gb(total_gb):
    """
    Eski sürümlerde Windows için toplam RAM'in %20'si sabit pay olarak ayrılıyordu; bu, RAM boşken bile büyük
    modelleri engelliyordu. Windows bekleyen/önbellek belleği kendisi boşaltır ve gerekirse sayfa dosyasını
    kullanır. Artık yalnızca küçük bir güvenlik payı kalır.
    """
    return 1.0


def ram_can_load(need_gb, total_gb=None):
    """
    (yüklenebilir_mi, kullanılabilir_GB, sıkışık_mı). Sınır: boş RAM + sayfa dosyasındaki boş yer (Windows'un
    gerçekten verebileceği bellek). Boş RAM'den fazlası gerekiyorsa yine yüklenir; Windows yer açar ('sıkışık').
    """
    avail = ram_available_gb()
    pf = pagefile_free_gb()
    usable = avail + max(0.0, pf) - ram_reserve_gb(total_gb or 0)
    return need_gb <= usable, max(0.0, usable), need_gb > avail


def detect_hardware():
    hw = Hardware()
    gpus = _query_nvidia_smi()
    if gpus:
        best = max(gpus, key=lambda g: g["vram_gb"])
        hw.vendor = "nvidia"
        hw.gpu_name = best["name"]
        hw.gpu_index = best["index"]
        hw.vram_gb = best["vram_gb"]
        hw.compute_cap = best["cc"]
        hw.driver = best.get("driver", "")
        try:
            hw.driver_major = int(hw.driver.split(".")[0])
        except (ValueError, IndexError):
            hw.driver_major = 0
    hw.ram_total_gb = ram_total_gb()
    hw.pagefile_gb = pagefile_gb()
    return hw


# =============================================================================
# Bellek Koruyucu — profiller
# =============================================================================
PROFILES = ["full", "normal", "balanced", "low", "cpu"]
PROFILE_ICON = {"full": "🟢", "normal": "🔵", "balanced": "🟡", "low": "🟠", "cpu": "⚪"}


def auto_profile(model, hw):
    """Modelin tahmini ihtiyacına ve karta göre en hızlı GÜVENLİ profili seçer."""
    if not hw.cuda_usable and hw.vendor != "nvidia":
        return "cpu"
    v = hw.vram_gb
    if v >= model["vram_full"]:
        return "full"
    if v >= model["vram_offload"]:
        return "normal"
    if v >= model["vram_offload"] * 0.8:
        return "balanced"
    if v >= 1.9:
        return "low"
    return "cpu"


def lower_profile(profile):
    i = PROFILES.index(profile)
    return PROFILES[i + 1] if i + 1 < len(PROFILES) else None


def model_ram_need_gb(model, profile, disk_gb=None):
    """Modeli yüklemek için gereken tahmini RAM (ağırlıklar RAM'den geçer)."""
    size = disk_gb if disk_gb else model["size_gb"]
    factor = 2.0 if (profile == "cpu" and model["dtype"] == "fp16") else 1.0  # CPU'da fp16 → fp32
    return size * factor + 1.5


def fit_level(model, hw):
    """'ok' | 'slow' | 'no' — model listesinde gösterilen uygunluk etiketi."""
    if model_ram_need_gb(model, "normal") > hw.ram_total_gb + max(0.0, hw.pagefile_gb) - ram_reserve_gb(hw.ram_total_gb):
        return "no"
    prof = auto_profile(model, hw)
    if prof in ("full", "normal"):
        return "ok"
    if prof in ("balanced", "low"):
        return "slow"
    return "slow" if model["size_gb"] <= 3 else "no"


class RamWatchdog:
    """İş sürerken RAM'i izler; kritik seviyeye inerse işi güvenle durdurur (sistem donmasın diye)."""

    def __init__(self, abort_event, total_gb, on_trigger=None, interval=1.0, confirm=3):
        self.abort_event = abort_event
        self.limit_gb = max(1.0, min(2.0, total_gb * 0.05))
        self.confirm = max(1, int(confirm))      # art arda bu kadar ölçümde düşük kalırsa (anlık düşüşler sayılmaz)
        self.on_trigger = on_trigger
        self.interval = interval
        self.triggered = False
        self.free_at_trigger = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def _critical(self):
        """
        Gerçekten tehlikeli mi? Boş RAM azaldığında Windows bekleyen belleği boşaltır ve sayfa dosyasına yazar;
        bu normaldir. Tehlike, sayfa dosyasında da yer kalmadığında (ya da sayfa dosyası yokken) başlar.
        """
        free = ram_available_gb()
        if free >= self.limit_gb:
            return False, free
        pf = pagefile_free_gb()
        if pf > 2.0:
            return False, free
        return True, free

    def _run(self):
        low = 0
        while not self._stop.wait(self.interval):
            crit, free = self._critical()
            low = low + 1 if crit else 0
            if low >= self.confirm:
                self.triggered = True
                self.free_at_trigger = free
                self.abort_event.set()
                if self.on_trigger:
                    try:
                        self.on_trigger(free)
                    except Exception:
                        pass
                return


def is_oom_error(exc):
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cublas_status_alloc_failed" in msg)


def free_gpu_memory():
    gc.collect()
    torch = sys.modules.get("torch")      # torch'u burada asla ilk kez içe aktarma
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            release_pinned_cache()
    except Exception:
        pass


# =============================================================================
# Boyut hesabı
# =============================================================================
ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9), "21:9": (21, 9)}
RES_SCALE = {"low": 0.75, "normal": 1.0, "high": 1.25}


def compute_size(base_px, aspect, orientation, res):
    """Modelin doğal çözünürlüğüne göre, 64'ün katı olan genişlik/yükseklik döndürür."""
    a, b = ASPECTS.get(aspect, (1, 1))
    if orientation == "portrait":
        a, b = b, a
    side = base_px * RES_SCALE.get(res, 1.0)
    area = side * side
    w = math.sqrt(area * a / b)
    h = area / w
    w = max(256, int(round(w / 64.0)) * 64)
    h = max(256, int(round(h / 64.0)) * 64)
    return w, h


def creativity_to_cfg(creativity):
    """1 = prompt'a çok sadık (yüksek CFG), 10 = serbest yorum (düşük CFG)."""
    c = min(10, max(1, int(creativity)))
    return round(11.5 - 0.9 * c, 1)


# =============================================================================
# Model kayıtları — YALNIZCA ticari kullanıma uygun ve onaysız (gated değil) modeller
# =============================================================================
IMAGE_MODELS = [
    {
        "key": "sd15", "name": "Stable Diffusion 1.5",
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "variant": "fp16",
        "size_gb": 2.1, "license": "CreativeML OpenRAIL-M", "dtype": "fp16",
        "base_px": 512, "steps": {"fast": 20, "balanced": 30, "detailed": 40, "ultra": 50},
        "cfg": True, "neg": True, "fixed_cfg": None,
        "vram_full": 3.5, "vram_offload": 2.2, "prompt_style": "tags",
        "samplers": True, "img2img": "StableDiffusionImg2ImgPipeline", "min_refine": 12,
        "detail_strength": {"face": 0.4, "hand": 0.45}, "hires_strength": 0.45, "hires_max_mp": 1.05,
        "ignore": ["safety_checker/*", "feature_extractor/*"], "desc": "desc_sd15",
        "load_kwargs": {"safety_checker": None, "requires_safety_checker": False, "feature_extractor": None},
    },
    {
        "key": "sdxl", "name": "Stable Diffusion XL 1.0",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0", "variant": "fp16",
        "size_gb": 6.9, "license": "CreativeML OpenRAIL++-M", "dtype": "fp16",
        "base_px": 1024, "steps": {"fast": 25, "balanced": 35, "detailed": 45, "ultra": 60},
        "cfg": True, "neg": True, "fixed_cfg": None,
        "vram_full": 9.0, "vram_offload": 6.0, "prompt_style": "tags",
        "samplers": True, "img2img": "StableDiffusionXLImg2ImgPipeline", "min_refine": 12,
        "detail_strength": {"face": 0.4, "hand": 0.45}, "hires_strength": 0.4, "hires_max_mp": 2.4,
        "ignore": ["vae_1_0/*", "vae_decoder/*", "vae_encoder/*", "text_encoder/*.onnx*", "unet/*.onnx*"],
        "desc": "desc_sdxl", "load_kwargs": {},
    },
    {
        "key": "zimage_turbo", "name": "Z-Image-Turbo",
        "repo": "Tongyi-MAI/Z-Image-Turbo", "variant": None,
        "size_gb": 20.5, "license": "Apache-2.0", "dtype": "bf16",
        "base_px": 1024, "steps": {"fast": 8, "balanced": 9, "detailed": 12, "ultra": 16},
        "cfg": False, "neg": False, "fixed_cfg": 0.0,
        "vram_full": 17.0, "vram_offload": 13.0, "prompt_style": "natural",
        "samplers": False, "turbo": 9, "img2img": "ZImageImg2ImgPipeline", "min_refine": 4,
        "detail_strength": {"face": 0.35, "hand": 0.4}, "hires_strength": 0.35, "hires_max_mp": 2.4,
        "ignore": [], "desc": "desc_zimage", "load_kwargs": {},
    },
    {
        "key": "flux2_klein_4b", "name": "FLUX.2 [klein] 4B",
        "repo": "black-forest-labs/FLUX.2-klein-4B", "variant": None,
        "size_gb": 16.0, "license": "Apache-2.0", "dtype": "bf16",
        "base_px": 1024, "steps": {"fast": 4, "balanced": 4, "detailed": 6, "ultra": 8},
        "cfg": False, "neg": False, "fixed_cfg": 1.0,
        "vram_full": 14.0, "vram_offload": 9.0, "prompt_style": "natural",
        "samplers": False, "turbo": 4, "img2img": "Flux2KleinInpaintPipeline", "min_refine": 3,
        "detail_strength": {"face": 0.35, "hand": 0.4}, "hires_strength": 0.35, "hires_max_mp": 2.4,
        "ignore": [], "desc": "desc_klein", "load_kwargs": {},
    },
]
IMAGE_MODEL_BY_KEY = {m["key"]: m for m in IMAGE_MODELS}


def model_family(model):
    """Modelin mimarisi: programın modellerinde anahtarın kendisi, kullanıcının eklediği modelde 'sd15' / 'sdxl'."""
    return model.get("family") or model["key"]

# Ollama LLM'leri — hepsi Apache-2.0 (Qwen2.5'in 3B/72B sürümleri farklı lisanslı olduğu için listede yok).
LLM_MODELS = [
    {"key": "qwen3:4b", "name": "Qwen3 4B", "size": "~2.5 GB", "size_gb": 2.5, "license": "Apache-2.0", "vram": 3.5, "desc": "desc_qwen3_4b"},
    {"key": "qwen3:1.7b", "name": "Qwen3 1.7B", "size": "~1.4 GB", "size_gb": 1.4, "license": "Apache-2.0", "vram": 2.0, "desc": "desc_qwen3_17b"},
    {"key": "qwen3:8b", "name": "Qwen3 8B", "size": "~5.2 GB", "size_gb": 5.2, "license": "Apache-2.0", "vram": 6.0, "desc": "desc_qwen3_8b"},
    {"key": "qwen2.5:7b", "name": "Qwen2.5 7B", "size": "~4.7 GB", "size_gb": 4.7, "license": "Apache-2.0", "vram": 5.5, "desc": "desc_qwen25_7b"},
]
LLM_MODEL_BY_KEY = {m["key"]: m for m in LLM_MODELS}
DEFAULT_LLM = "qwen3:4b"

UPSCALER = {
    "key": "realesrgan_x2", "name": "Real-ESRGAN x2plus",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "file": "RealESRGAN_x2plus.pth", "size": "~64 MB", "size_gb": 0.07, "license": "BSD-3-Clause", "desc": "desc_esrgan",
}


QUALITIES = ("fast", "balanced", "detailed", "ultra")
# Örnekleyiciler yalnızca SD 1.5 / SDXL için seçilebilir; turbo/flow modeller kendi zamanlayıcısını kullanır.
SAMPLERS = ("auto", "dpmpp_2m_karras", "euler_a", "euler", "unipc", "default")
SAMPLER_NAMES = {"dpmpp_2m_karras": "DPM++ 2M Karras", "euler_a": "Euler a", "euler": "Euler",
                 "unipc": "UniPC", "default": "PNDM / Euler (model)"}
MAX_STEPS = 150


def resolve_steps(model, quality, manual=0):
    """Elle girilen adım sayısı (>0) her zaman önceliklidir; yoksa kalite ön ayarı."""
    try:
        manual = int(manual or 0)
    except (TypeError, ValueError):
        manual = 0
    if manual > 0:
        return max(1, min(MAX_STEPS, manual))
    q = quality if quality in model["steps"] else "balanced"
    return int(model["steps"][q])


def refine_steps(model, steps, strength):
    """
    img2img'de gerçekte atılan adım ≈ adım × güç. Düzeltme/ikinci geçişin yeterli adım alması için
    toplam adım, en az 'min_refine' gerçek adım verecek kadar artırılır.
    """
    need = int(math.ceil(model.get("min_refine", 8) / max(0.05, float(strength))))
    return max(1, min(MAX_STEPS, max(int(steps), need)))


def hires_enabled(mode, model, quality, profile):
    """İkinci geçiş (hires fix) açık mı? auto: SD 1.5'te dengeli ve üstü, SDXL'de yalnızca Ultra (yeterli VRAM'de)."""
    if mode == "off" or not model.get("img2img"):
        return False
    if mode == "on":
        return True
    if profile == "cpu":
        return False
    if model_family(model) == "sd15":
        return quality in ("balanced", "detailed", "ultra")
    if model_family(model) == "sdxl":
        return quality == "ultra" and profile in ("full", "normal")
    return False


def hires_size(w, h, model, scale=1.5):
    """İkinci geçişin hedef boyutu (16'nın katı, modelin megapiksel sınırı içinde). Değmezse None."""
    s = float(scale)
    max_px = float(model.get("hires_max_mp", 1.0)) * 1_000_000
    if w * h * s * s > max_px:
        s = math.sqrt(max_px / float(w * h))
    if s < 1.15:
        return None
    return max(64, int(round(w * s / 16.0)) * 16), max(64, int(round(h * s / 16.0)) * 16)


def recommended_model(hw):
    """Bu PC için önerilen model: rahat çalışanların en kalitelisi, yoksa en hafifi."""
    order = ["zimage_turbo", "flux2_klein_4b", "sdxl", "sd15"]
    for key in order:
        if fit_level(IMAGE_MODEL_BY_KEY[key], hw) == "ok":
            return IMAGE_MODEL_BY_KEY[key]
    if hw.vendor == "nvidia" and fit_level(IMAGE_MODEL_BY_KEY["sdxl"], hw) == "slow":
        return IMAGE_MODEL_BY_KEY["sdxl"]
    return IMAGE_MODEL_BY_KEY["sd15"]


# =============================================================================
# Görsel modelleri: indirme ve silme (PyTorch gerekmez — sadece huggingface_hub)
# =============================================================================
MARKER = ".ails_complete.json"


def image_model_dir(models_root, model):
    if model.get("custom"):
        return model["file"]          # kullanıcının modeli tek dosyadır
    return os.path.join(models_root, "image", model["key"])


def upscaler_path(models_root):
    return os.path.join(models_root, "upscale", UPSCALER["file"])


def is_image_model_installed(models_root, model):
    if model.get("custom"):
        return os.path.isfile(model["file"])
    return os.path.exists(os.path.join(image_model_dir(models_root, model), MARKER))


def is_upscaler_installed(models_root):
    p = upscaler_path(models_root)
    return os.path.exists(p) and os.path.getsize(p) > 10 * MB


def installed_size_gb(models_root, model):
    d = image_model_dir(models_root, model)
    try:
        with open(os.path.join(d, MARKER), encoding="utf-8") as f:
            return float(json.load(f).get("size_gb", 0)) or model["size_gb"]
    except Exception:
        return model["size_gb"]


def allow_patterns(model):
    weights = "*/*.fp16.safetensors" if model.get("variant") == "fp16" else "*/*.safetensors"
    # Sadece diffusers alt klasörleri: kökteki tek parça dev .safetensors/.ckpt dosyaları indirilmez.
    return ["model_index.json", "*/*.json", "*/*.txt", "*/*.model", "*/*.jinja", "*/*.tiktoken", weights]


def ignore_patterns(model):
    return ["*.ckpt", "*.bin", "*.pt", "*.pth", "*.onnx", "*.onnx_data", "*.msgpack", "*.h5"] + list(model.get("ignore", []))


def expected_download_bytes(model):
    """Gerçek indirme boyutunu Hugging Face'ten öğrenir (bilinmezse None)."""
    try:
        from huggingface_hub import snapshot_download
        infos = snapshot_download(model["repo"], allow_patterns=allow_patterns(model),
                                  ignore_patterns=ignore_patterns(model), dry_run=True)
        return sum((i.file_size or 0) for i in infos)
    except Exception:
        return None


def download_image_model(models_root, model, progress_cb, abort_event):
    """
    Modeli indirir. progress_cb(done_bytes, total_bytes) her saniye çağrılır.
    Yarıda kalan indirme bir sonraki denemede kaldığı yerden devam eder.
    """
    from huggingface_hub import snapshot_download
    dest = image_model_dir(models_root, model)
    total = expected_download_bytes(model) or int(model["size_gb"] * GB)
    already = folder_size_bytes(dest) if os.path.isdir(dest) else 0
    check_space(models_root, max(0, total - already) / GB + 1.0)
    os.makedirs(dest, exist_ok=True)
    result = {"err": None}

    def _run():
        try:
            snapshot_download(model["repo"], local_dir=dest, allow_patterns=allow_patterns(model),
                              ignore_patterns=ignore_patterns(model), max_workers=4)
        except Exception as e:  # noqa
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(1.0)
        if progress_cb:
            progress_cb(folder_size_bytes(dest), total)
        if abort_event is not None and abort_event.is_set():
            # huggingface_hub indirmeyi iptal etmeyi desteklemiyor; kısmi dosyalar bir sonraki
            # indirmede kaldığı yerden devam ettirilir.
            raise AbortedError("user")
    if result["err"] is not None:
        err = result["err"]
        if isinstance(err, OSError) and getattr(err, "errno", None) == 28:
            raise DiskFullError(drive_label(models_root), total / GB, disk_free_gb(models_root))
        raise err
    if not os.path.exists(os.path.join(dest, "model_index.json")):
        raise RuntimeError("model_index.json missing — download incomplete")
    size_gb = folder_size_bytes(dest) / GB
    with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
        json.dump({"repo": model["repo"], "size_gb": round(size_gb, 2), "date": time.strftime("%Y-%m-%d %H:%M")}, f)
    return dest


def download_file(url, dest_path, progress_cb, abort_event, timeout=60, need_gb=None, sha256=None):
    """Dosyayı .part olarak indirir; sha256 verilmişse içerik doğrulanmadan yerine konmaz."""
    import hashlib
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    hasher = hashlib.sha256() if sha256 else None
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            check_space(dest_path, (need_gb if need_gb else total / GB) + 0.5)
            done = 0
            last = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if abort_event is not None and abort_event.is_set():
                        raise AbortedError("user")
                    if chunk:
                        f.write(chunk)
                        if hasher is not None:
                            hasher.update(chunk)
                        done += len(chunk)
                        if progress_cb and time.time() - last > 0.5:
                            last = time.time()
                            progress_cb(done, total)
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            raise DiskFullError(drive_label(dest_path), (need_gb or 1.0), disk_free_gb(dest_path)) from e
        raise
    if hasher is not None and hasher.hexdigest().lower() != sha256.lower():
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise NetworkError("downloaded file is corrupted (checksum mismatch): " + os.path.basename(dest_path))
    os.replace(tmp, dest_path)
    return dest_path


def delete_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


# =============================================================================
# Programın kendi Python ortamı (venv) ve görsel motoru kurulumu
# Genel Python'a hiçbir şey kurulmaz; her şey veri klasöründeki runtime\venv içine gider.
# =============================================================================
BASIC_MODULES = {"PIL": "pillow", "psutil": "psutil", "requests": "requests", "numpy": "numpy",
                 "huggingface_hub": "huggingface_hub"}
ENGINE_PIP = ["diffusers>=0.37", "transformers", "accelerate", "safetensors", "sentencepiece", "protobuf", "peft"]
REQUIRED_PIPELINES = ["ZImagePipeline", "Flux2KleinPipeline", "StableDiffusionXLPipeline"]
ENGINE_NEED_GB = {"cuda": 9.0, "cpu": 3.0}
ENGINE_SIZE = {"cuda": ("~3 GB", "~6 GB"), "cpu": ("~0.5 GB", "~2 GB")}


def engine_kind(hw):
    return "cuda" if hw.vendor == "nvidia" else "cpu"


def running_in_venv(paths):
    try:
        return os.path.normcase(os.path.realpath(sys.prefix)) == os.path.normcase(os.path.realpath(paths.venv))
    except Exception:
        return False


def base_python():
    """venv'i oluşturacak / programı yeniden başlatacak asıl Python yorumlayıcısı."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        for cand in (os.path.join(base, "python", "python.exe"), os.path.join(base, "runtime", "python.exe")):
            if os.path.exists(cand):
                return cand
        return None
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):      # bir venv içindeyiz
        be = getattr(sys, "_base_executable", None)
        if be and os.path.exists(be) and \
                os.path.normcase(os.path.dirname(os.path.abspath(be))) != os.path.normcase(os.path.dirname(os.path.abspath(sys.executable))):
            return be
        try:
            with open(os.path.join(sys.prefix, "pyvenv.cfg"), encoding="utf-8") as f:
                for line in f:
                    k, _, v = line.partition("=")
                    if k.strip() == "home":
                        exe = os.path.join(v.strip(), "python.exe" if IS_WIN else "python3")
                        if os.path.exists(exe):
                            return exe
        except Exception:
            pass
    return sys.executable


def venv_ok(paths):
    if not os.path.exists(paths.venv_python):
        return False
    try:
        r = subprocess.run([paths.venv_python, "-c", "import sys, pip; print(sys.version_info[0])"],
                           capture_output=True, text=True, timeout=120, env=sub_env(paths), **no_window_kwargs())
        return r.returncode == 0
    except Exception:
        return False


def create_venv(paths, log_cb=None, python_exe=None):
    py = python_exe or base_python()
    if not py:
        raise RuntimeError("Python runtime not found")
    shutil.rmtree(paths.venv, ignore_errors=True)
    os.makedirs(paths.runtime, exist_ok=True)
    setup_log(paths, "venv: %s -m venv %s" % (py, paths.venv))
    r = subprocess.run([py, "-m", "venv", paths.venv], capture_output=True, text=True, timeout=900,
                       env=sub_env(paths), **no_window_kwargs())
    if r.returncode != 0 or not os.path.exists(paths.venv_python):
        detail = (r.stderr or r.stdout or "").strip()[-600:]
        setup_log(paths, "venv failed (code %s):\n%s" % (r.returncode, (r.stderr or "") + (r.stdout or "")))
        if "ensurepip" in detail.lower() or "venv" in detail.lower():
            raise RuntimeError("NO_VENV: " + detail)
        raise RuntimeError(detail or "venv creation failed")
    if not venv_ok(paths):
        r2 = subprocess.run([paths.venv_python, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True,
                            timeout=600, env=sub_env(paths), **no_window_kwargs())
        if not venv_ok(paths):
            setup_log(paths, "ensurepip failed:\n" + (r2.stderr or "") + (r2.stdout or ""))
            raise RuntimeError("NO_VENV: pip is missing in the new environment")
    if log_cb:
        log_cb("venv: " + paths.venv)
    return True


def missing_basic(paths):
    if not os.path.exists(paths.venv_python):
        return list(BASIC_MODULES.values())
    code = ("import importlib.util as u, json; "
            "print(json.dumps([m for m in %r if u.find_spec(m) is None]))" % (list(BASIC_MODULES),))
    try:
        r = subprocess.run([paths.venv_python, "-c", code], capture_output=True, text=True, timeout=120,
                           env=sub_env(paths), **no_window_kwargs())
        mods = json.loads(r.stdout.strip().splitlines()[-1])
        return [BASIC_MODULES[m] for m in mods]
    except Exception:
        return list(BASIC_MODULES.values())


# -----------------------------------------------------------------------------
# İlk kurulum günlüğü (logs\setup.log) — ilk açılışta henüz ana program yok; her şey buraya yazılır
# -----------------------------------------------------------------------------
_SETUP_LOG_LOCK = threading.Lock()


def setup_log_path(paths):
    return os.path.join(paths.logs, "setup.log")


def setup_log(paths, text):
    """Kurulum günlüğüne yazar; asla hata fırlatmaz."""
    try:
        os.makedirs(paths.logs, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with _SETUP_LOG_LOCK, open(setup_log_path(paths), "a", encoding="utf-8", errors="replace") as f:
            for line in (str(text).splitlines() or [""]):
                f.write(f"[{stamp}] {line}\n")
    except Exception:
        pass


SUPPORTED_PY = ((3, 10), (3, 15))   # motor paketlerinin (PyTorch vb.) hazır paketi olan Python sürümleri
PORTABLE_PY = "3.12"


def python_report():
    import platform
    import struct
    exe = sys.executable or ""
    return "\n".join([
        "Python %s (%d-bit) %s" % (platform.python_version(), struct.calcsize("P") * 8, platform.machine()),
        "executable: " + exe,
        "base_prefix: " + str(getattr(sys, "base_prefix", "")),
        "OS: " + platform.platform(),
        "store/managed python: %s" % is_store_python()])


def is_store_python():
    exe = (sys.executable or "").lower().replace("/", "\\")
    return "\\windowsapps\\" in exe or "pythonsoftwarefoundation" in exe


def system_python_problem():
    """Bilgisayardaki Python programın ortamı için uygun değilse nedeni: 'old' | 'new' | '32bit' | None."""
    import struct
    if struct.calcsize("P") * 8 != 64:
        return "32bit"
    v = tuple(sys.version_info[:2])
    if v < SUPPORTED_PY[0]:
        return "old"
    if v > SUPPORTED_PY[1]:
        return "new"
    return None


# -----------------------------------------------------------------------------
# Programa özel Python (isteğe bağlı, onayla): uv aracı PyPI'den, Python 3.12 'python-build-standalone'
# derlemesi uv ile indirilir. Hepsi veri klasöründeki runtime\ içine gider; kayıt defterine, PATH'e ya da
# kullanıcı klasörüne bir şey yazılmaz. Böylece bilgisayardaki Python'un sürümü / Microsoft Store güncellemeleri
# programın ortamını bozamaz.
# -----------------------------------------------------------------------------
UV_WHEELS = {
    "amd64": {"version": "0.13.9", "size": 15662317,
              "url": "https://files.pythonhosted.org/packages/dd/ef/ceeabf3631117f7354787d9fb1ffe943899e59ed4b887a696ed"
                     "547e47e76/uv-0.13.9-py3-none-win_amd64.whl",
              "sha256": "6a44b2ed8111d2f60d2b92627c5dccda3f7ed7570b18ce115e569cd68c28246b"},
    "arm64": {"version": "0.13.9", "size": 15841844,
              "url": "https://files.pythonhosted.org/packages/da/d1/d97a5db1f27e077e4d6329223f6b3f719c02c97e4092043f13e"
                     "628558d05/uv-0.13.9-py3-none-win_arm64.whl",
              "sha256": "eeb92051c2123b9730c46b04cc843cfa4bc7d7b0aec56df431a2215ef4cb0273"},
}
UV_LICENSE = "MIT / Apache-2.0"
PORTABLE_SIZE_MB = 45            # uv ~16 MB + Python 3.12 ~25–30 MB
PORTABLE_LICENSE = "PSF-2.0"


def portable_root(paths):
    return os.path.join(paths.runtime, "python")


def uv_exe_path(paths):
    return os.path.join(paths.runtime, "tools", "uv.exe" if IS_WIN else "uv")


def _ver_key(path):
    return [int(x) for x in re.findall(r"\d+", os.path.basename(os.path.dirname(path)) if IS_WIN else
                                       os.path.basename(os.path.dirname(os.path.dirname(path))))]


def portable_python_exe(paths):
    """Programa özel Python kuruluysa yorumlayıcısının yolu, değilse None."""
    root = portable_root(paths)
    if IS_WIN:
        found = glob.glob(os.path.join(root, "cpython-%s*" % PORTABLE_PY, "python.exe"))
    else:
        found = glob.glob(os.path.join(root, "cpython-%s*" % PORTABLE_PY, "bin", "python3"))
    found = [p for p in found if os.path.isfile(p) and "freethreaded" not in p.lower()]
    return sorted(found, key=_ver_key)[-1] if found else None


_PY_CHECK = "import sys, venv, ensurepip, ssl, sqlite3, tkinter; print('AILS_PY ' + sys.version.split()[0])"


def check_python(exe, paths=None, timeout=180):
    """Bir Python yorumlayıcısı programın ortamını kurabilir mi? Dönüş: (tamam, sürüm|hata)."""
    try:
        r = subprocess.run([exe, "-c", _PY_CHECK], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=sub_env(paths), **no_window_kwargs())
    except Exception as e:
        return False, str(e)[:300]
    for line in (r.stdout or "").splitlines():
        if line.startswith("AILS_PY "):
            return True, line[8:].strip()
    return False, ((r.stderr or "") + (r.stdout or "")).strip()[-400:]


def fetch_url(url, dest_path, sha256=None, progress_cb=None, abort_event=None, timeout=60, expected_size=0):
    """Yalnızca standart kütüphaneyle indirir (ilk kurulumda 'requests' henüz yok). sha256 tutmazsa dosya
    kullanılmaz."""
    import hashlib
    import urllib.error
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    hasher = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": APP_UA})
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or expected_size or 0)
            check_space(dest_path, total / GB + 0.2)
            last = 0.0
            while True:
                if abort_event is not None and abort_event.is_set():
                    raise AbortedError("user")
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                if progress_cb and time.time() - last > 0.3:
                    last = time.time()
                    progress_cb(done, total)
    except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise NetworkError(str(e)[:300]) from e
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        if getattr(e, "errno", None) == 28:
            raise DiskFullError(drive_label(dest_path), (expected_size or done) / GB + 0.2,
                                disk_free_gb(dest_path)) from e
        raise
    if sha256 and hasher.hexdigest().lower() != sha256.lower():
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise NetworkError("downloaded file is corrupted (checksum mismatch): " + os.path.basename(dest_path))
    os.replace(tmp, dest_path)
    return dest_path


def _uv_arch():
    import platform
    return "arm64" if (platform.machine() or "").lower() in ("arm64", "aarch64") else "amd64"


def install_uv(paths, progress_cb=None, abort_event=None, log_cb=None):
    """uv aracını PyPI'deki resmî paketinden (sha256 doğrulamalı) runtime\\tools içine çıkarır."""
    import zipfile
    exe = uv_exe_path(paths)
    if os.path.exists(exe):
        return exe
    w = UV_WHEELS[_uv_arch()]
    whl = os.path.join(paths.tmp, os.path.basename(w["url"]))
    if log_cb:
        log_cb("   uv %s (%s) — %.0f MB" % (w["version"], UV_LICENSE, w["size"] / MB))
    setup_log(paths, "uv: " + w["url"])
    fetch_url(w["url"], whl, w["sha256"], progress_cb, abort_event, expected_size=w["size"])
    try:
        with zipfile.ZipFile(whl) as z:
            want = "/scripts/uv.exe" if IS_WIN else "/scripts/uv"
            member = next((n for n in z.namelist() if n.lower().endswith(want)), None)
            if not member:
                raise RuntimeError("uv.exe not found in the package")
            os.makedirs(os.path.dirname(exe), exist_ok=True)
            with z.open(member) as src, open(exe + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(exe + ".part", exe)
        if not IS_WIN:
            os.chmod(exe, 0o755)
    finally:
        try:
            os.remove(whl)
        except OSError:
            pass
    return exe


def uv_env(paths):
    """uv yalnızca veri klasörünü kullansın: kullanıcının uv ayarları/klasörleri, kayıt defteri ve PATH'e dokunulmaz."""
    env = sub_env(paths)
    for k in list(env):
        if k.startswith("UV_"):
            env.pop(k)
    root = portable_root(paths)
    env.update({"UV_PYTHON_INSTALL_DIR": root, "UV_PYTHON_BIN_DIR": os.path.join(root, "bin"),
                "UV_PYTHON_INSTALL_BIN": "0", "UV_PYTHON_INSTALL_REGISTRY": "0",
                "UV_CACHE_DIR": os.path.join(paths.cache, "uv"), "UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1",
                "UV_NATIVE_TLS": "1", "UV_PYTHON_PREFERENCE": "only-managed"})
    return env


def _run_uv(paths, uv, args, log_cb, abort_event=None, timeout=1800):
    proc = subprocess.Popen([uv] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=uv_env(paths), **no_window_kwargs())
    lines = []
    start = time.time()
    killer = threading.Timer(timeout, proc.kill)
    killer.daemon = True
    killer.start()
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                lines.append(line)
                if log_cb and len(lines) <= 40:
                    log_cb("   uv › " + line[:170])
            if abort_event is not None and abort_event.is_set():
                proc.kill()
                raise AbortedError("user")
        proc.wait()
    finally:
        killer.cancel()
        if proc.poll() is None:
            proc.kill()
    out = "\n".join(lines)
    setup_log(paths, "uv %s → code %s (%.0f s)\n%s" % (" ".join(args), proc.returncode, time.time() - start, out[-6000:]))
    return proc.returncode, out


def install_portable_python(paths, log_cb=None, progress_cb=None, abort_event=None):
    """Programa özel Python 3.12'yi kurar (gerekirse önce uv). Dönüş: python.exe yolu."""
    log = log_cb or (lambda s: None)
    check_space(paths.root, 0.4)
    exe = portable_python_exe(paths)
    if exe and check_python(exe, paths)[0]:
        return exe
    uv = install_uv(paths, progress_cb, abort_event, log)
    if progress_cb:
        progress_cb(-1, 0)
    args = ["python", "install", PORTABLE_PY, "--install-dir", portable_root(paths)]
    code, out = _run_uv(paths, uv, args, log, abort_event)
    if code != 0 and "unexpected argument" in out.lower():
        code, out = _run_uv(paths, uv, ["python", "install", PORTABLE_PY], log, abort_event)
    if code != 0:
        low = out.lower()
        if any(h in low for h in ("dns", "connect", "timed out", "network", "tcp", "unreachable", "failed to fetch",
                                  "failed to download")):
            raise NetworkError("uv: " + pip_error_summary(out, 3, 300))
        raise InstallError("Python %s could not be installed." % PORTABLE_PY, pip_error_summary(out), None, "error")
    exe = portable_python_exe(paths)
    if not exe:
        raise InstallError("Python %s could not be installed." % PORTABLE_PY, "python.exe not found in " +
                           portable_root(paths), None, "error")
    ok, info = check_python(exe, paths)
    setup_log(paths, "portable python: %s → %s %s" % (exe, ok, info))
    if not ok:
        raise InstallError("Python %s is incomplete." % PORTABLE_PY, info, None, "error")
    return exe


def create_venv_uv(paths, python_exe, log_cb=None):
    """'python -m venv' olmazsa (ör. eksik venvlauncher) ortamı uv ile kurar; pip de eklenir (--seed)."""
    uv = uv_exe_path(paths)
    if not os.path.exists(uv):
        raise RuntimeError("uv not available")
    shutil.rmtree(paths.venv, ignore_errors=True)
    code, out = _run_uv(paths, uv, ["venv", paths.venv, "--python", python_exe, "--seed"], log_cb)
    if code != 0 or not venv_ok(paths):
        raise RuntimeError("NO_VENV: " + pip_error_summary(out, 4, 400))
    return True


def fix_venv_home(new_root, old_root):
    """Veri klasörü taşındıysa: programa özel Python'u gösteren venv ayarını (pyvenv.cfg) yeni yere çevirir.
    Böylece taşıma sonrası motorun (PyTorch vb.) yeniden indirilmesi gerekmez."""
    cfg = os.path.join(new_root, "runtime", "venv", "pyvenv.cfg")
    try:
        with open(cfg, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return False
    old_abs, new_abs = os.path.abspath(old_root), os.path.abspath(new_root)
    old_n = os.path.normcase(old_abs)
    changed = False
    out = []
    for line in lines:
        k, sep, v = line.partition("=")
        key, val = k.strip(), v.strip()
        if sep and val:
            new_val = val
            if key == "command":
                new_val = val.replace(old_abs, new_abs)
            elif os.path.normcase(os.path.abspath(val)).startswith(old_n + os.sep):
                new_val = os.path.join(new_abs, os.path.relpath(os.path.abspath(val), old_abs))
            if new_val != val:
                line = "%s = %s" % (key, new_val)
                changed = True
        out.append(line)
    if changed:
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return changed


def install_basics(paths, pkgs, log_cb, progress_cb=None):
    """Programın açılması için gereken temel paketler. Önce pip güncellenir; hazır paket (wheel) tercih edilir,
    derleme gerektiren kaynak paketlere gidilmez. Antivirüs dosyayı kilitlediyse bir kez daha denenir."""
    st, out = run_pip(paths, ["--upgrade", "pip"], lambda s: None)
    setup_log(paths, "pip upgrade: " + st)
    args = ["--prefer-binary"] + list(pkgs)
    status, out = run_pip(paths, args, log_cb, None, progress_cb)
    if status in ("error",) and pip_hint(out) in ("access", None):
        setup_log(paths, "pip failed once, retrying:\n" + out[-4000:])
        time.sleep(3.0)
        status, out = run_pip(paths, args, log_cb, None, progress_cb)
    if status != "ok":
        setup_log(paths, "pip install %s → %s\n%s" % (" ".join(pkgs), status, out[-20000:]))
    raise_for_pip(status, paths, 1.0, out)
    return True


_NOSPACE_HINTS = ("errno 28", "no space left", "winerror 112", "not enough space")
_NETWORK_HINTS = ("retrying (retry(", "connection broken", "could not fetch url", "max retries exceeded",
                  "read timed out", "connecttimeout", "readtimeout", "connectionerror", "incompleteread",
                  "connection reset", "connection aborted", "name resolution", "getaddrinfo failed",
                  "temporary failure", "network is unreachable", "proxyerror", "ssl: ", "sslerror")
_NOMATCH_HINTS = ("no matching distribution", "could not find a version")


def classify_pip_output(text):
    low = (text or "").lower()
    if any(h in low for h in _NOSPACE_HINTS):
        return "nospace"
    if any(h in low for h in _NETWORK_HINTS):
        return "network"     # pip internet hatasında da sonunda "no matching distribution" der
    if any(h in low for h in _NOMATCH_HINTS):
        return "nomatch"
    return "error"


_DL_RE = re.compile(r"^\s*Downloading\s+(\S+)\s+\(([\d.]+)\s*(kB|MB|GB)\)", re.I)
_UNIT = {"kb": 1024, "mb": MB, "gb": GB}


def run_pip(paths, args, log_cb, abort_event=None, progress_cb=None):
    """
    venv içinde pip çalıştırır. Dönüş: (durum, çıktı). durum: ok | nospace | network | nomatch | error
    progress_cb(done, total): büyük paket indirilirken geçici klasörün büyümesinden hesaplanır;
    kurulum aşamasında progress_cb(-1, 0) (belirsiz ilerleme) çağrılır.
    """
    os.makedirs(paths.tmp, exist_ok=True)
    cmd = [paths.venv_python, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
           "--no-input", "--progress-bar", "off"] + list(args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=sub_env(paths), **no_window_kwargs())
    lines = []
    state = {"total": 0, "base": 0, "active": False}
    stop = threading.Event()

    def monitor():
        while not stop.wait(1.0):
            if state["active"] and progress_cb and state["total"]:
                done = max(0, folder_size_bytes(paths.tmp) - state["base"])
                progress_cb(min(done, state["total"]), state["total"])

    mt = threading.Thread(target=monitor, daemon=True)
    mt.start()
    last_logged = ""
    retry_logged = False
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            lines.append(line)
            if len(lines) > 600:
                del lines[:100]
            if abort_event is not None and abort_event.is_set():
                proc.kill()
                raise AbortedError("user")
            m = _DL_RE.match(line)
            if m:
                size = float(m.group(2)) * _UNIT[m.group(3).lower()]
                if size >= 20 * MB:
                    state.update(total=int(size), base=folder_size_bytes(paths.tmp), active=True)
            if line.startswith("Installing collected packages"):
                state["active"] = False
                if progress_cb:
                    progress_cb(-1, 0)
            show = line.strip()
            if not show:
                continue
            if show.startswith("WARNING: Retrying"):
                if not retry_logged:
                    retry_logged = True
                    log_cb("   pip › " + show[:150])
                continue
            if show.startswith(("Collecting", "Downloading", "Installing", "Successfully", "ERROR")):
                if show != last_logged and "metadata" not in show:
                    log_cb("   pip › " + show[:170])
                    last_logged = show
        proc.wait()
    finally:
        stop.set()
        if proc.poll() is None:
            proc.kill()
    out = "\n".join(lines)
    if proc.returncode == 0:
        return "ok", out
    return classify_pip_output(out), out


class InstallError(RuntimeError):
    """pip başarısız oldu. detail: pip'in hata satırları; hint: olası neden (longpath/access/build/nomatch/...)."""

    def __init__(self, message, detail="", hint=None, status="error"):
        super().__init__(message)
        self.detail = detail
        self.hint = hint
        self.status = status


def pip_error_summary(out, max_lines=8, max_chars=900):
    """pip çıktısından kullanıcıya gösterilecek asıl hata satırları."""
    lines = [ln.rstrip() for ln in (out or "").splitlines() if ln.strip()]
    keep = [ln.strip() for ln in lines if re.match(r"\s*(ERROR|error|HINT|hint|note)\b", ln) or
            re.search(r"(Error|Exception|WinError|Errno|denied|not found|No matching)", ln)]
    if not keep:
        keep = [ln.strip() for ln in lines[-max_lines:]]
    seen, outl = set(), []
    for ln in keep[-(max_lines * 3):]:
        if ln not in seen:
            seen.add(ln)
            outl.append(ln[:240])
    text = "\n".join(outl[-max_lines:])
    return text[-max_chars:]


def pip_hint(out):
    """Bilinen kurulum sorunlarını tanır: longpath | access | build | nomatch | ssl | None."""
    low = (out or "").lower()
    if "long path" in low or "longpath" in low or "path too long" in low or "filename too long" in low:
        return "longpath"
    if any(k in low for k in ("winerror 5", "winerror 32", "access is denied", "permission denied",
                              "being used by another process", "erişim engellendi")):
        return "access"
    if any(k in low for k in ("microsoft visual c++", "unknown compiler", "failed building wheel",
                              "metadata-generation-failed", "subprocess-exited-with-error", "error: command",
                              "could not build wheels")):
        return "build"
    if any(h in low for h in _NOMATCH_HINTS) or "requires-python" in low:
        return "nomatch"
    if "certificate verify failed" in low or "sslcertverificationerror" in low:
        return "ssl"
    return None


def raise_for_pip(status, paths, need_gb, out=None):
    if status == "ok":
        return
    if status == "nospace":
        raise DiskFullError(drive_label(paths.root), need_gb, disk_free_gb(paths.root))
    if status == "network" and pip_hint(out) != "ssl":
        raise NetworkError("pip download failed")
    detail, hint = pip_error_summary(out), pip_hint(out)
    if status == "nomatch":
        raise InstallError("No compatible package was found for this computer.", detail, hint or "nomatch", status)
    raise InstallError("Installation failed (details in the log).", detail, hint, status)


def torch_index_urls(hw):
    """
    Kartın nesline ve sürücü sürümüne göre denenecek PyTorch paket kaynakları (sırayla).
    Sonraki kaynağa SADECE 'uyumlu paket yok' durumunda geçilir; disk/ağ hatasında durulur.
    """
    if hw.vendor != "nvidia":
        return ["https://download.pytorch.org/whl/cpu"]
    cc, drv = hw.compute_cap or 0.0, hw.driver_major or 0
    names = []
    if cc >= 12.0:                       # RTX 50 (Blackwell): en az CUDA 12.8
        if drv >= 580 or drv == 0:
            names.append("cu130")
        names += ["cu128", "cu129"]
    elif cc >= 7.0 or cc == 0:           # Volta / Turing / Ampere / Ada
        if drv >= 580:
            names.append("cu130")        # CUDA 13 için R580+ sürücü gerekir
        names += ["cu126", "cu128"]
    else:                                # Maxwell / Pascal
        names += ["cu126", "cu118"]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(f"https://download.pytorch.org/whl/{n}")
    return out


_VERIFY_CODE = r'''
import json
r = {}
try:
    import torch
    r["torch"] = torch.__version__
    r["cuda_build"] = torch.version.cuda
    r["cuda_ok"] = bool(torch.cuda.is_available())
    if r["cuda_ok"]:
        try:
            r["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
except BaseException as e:
    r["err"] = (type(e).__name__ + ": " + str(e))[:500]
else:
    try:
        import diffusers
        r["diffusers"] = diffusers.__version__
        r["pipes"] = all(hasattr(diffusers, p) for p in PIPES)
    except BaseException as e:
        r["derr"] = (type(e).__name__ + ": " + str(e))[:500]
print("AILS_VERIFY " + json.dumps(r))
'''.replace("PIPES", repr(REQUIRED_PIPELINES))


def verify_engine(paths, timeout=900):
    """Motoru AYRI bir süreçte sınar: ana süreç bozuk bir torch'la asla kirlenmez."""
    if not os.path.exists(paths.venv_python):
        return {"err": "venv missing"}
    try:
        r = subprocess.run([paths.venv_python, "-c", _VERIFY_CODE], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=sub_env(paths), **no_window_kwargs())
    except subprocess.TimeoutExpired:
        return {"err": "verification timed out"}
    except Exception as e:
        return {"err": str(e)}
    for line in (r.stdout or "").splitlines()[::-1]:
        if line.startswith("AILS_VERIFY "):
            try:
                return json.loads(line[len("AILS_VERIFY "):])
            except Exception:
                break
    return {"err": ((r.stderr or "") + (r.stdout or "")).strip()[-500:] or "verification failed"}


def engine_status(paths):
    """Motor kuruluysa işaret dosyasındaki bilgiler, değilse None (torch içe aktarılmaz)."""
    try:
        with open(paths.engine_marker, encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return None
    sp = paths.site_packages()
    if not (os.path.exists(os.path.join(sp, "torch", "__init__.py")) and os.path.isdir(os.path.join(sp, "diffusers"))):
        return None
    return info


def remove_engine_marker(paths):
    try:
        os.remove(paths.engine_marker)
    except OSError:
        pass


def remove_torch(paths):
    """Yarım/bozuk PyTorch kalıntılarını venv'den tamamen siler."""
    sp = paths.site_packages()
    if not os.path.isdir(sp):
        return
    for name in os.listdir(sp):
        low = name.lower()
        if low in ("torch", "torchgen", "functorch") or re.match(r"torch-.*\.dist-info$", low):
            shutil.rmtree(os.path.join(sp, name), ignore_errors=True)


def install_engine(paths, hw, log_cb, msg, progress_cb=None, abort_event=None):
    """
    Görsel motorunu (PyTorch + diffusers) programın kendi ortamına kurar ve doğrular.
    Sıra: yer kontrolü → (gerekirse bozuk torch temizliği) → doğru CUDA derlemesi → doğrulama →
    diffusers vb. (torch'a dokunmadan) → tekrar doğrulama → işaret dosyası.
    """
    kind = engine_kind(hw)
    need = ENGINE_NEED_GB[kind]
    check_space(paths.root, need)
    if not venv_ok(paths):
        raise EngineBrokenError("program environment is damaged")
    remove_engine_marker(paths)
    try:
        v = verify_engine(paths)
        torch_good = bool(v.get("torch")) and (kind == "cpu" or bool(v.get("cuda_build")))
        used_index = None
        if not torch_good:
            if v.get("torch") or os.path.isdir(os.path.join(paths.site_packages(), "torch")):
                remove_torch(paths)
            status, _out = "nomatch", ""
            for idx in torch_index_urls(hw):
                log_cb(msg("log_torch_index", idx=idx.rsplit("/", 1)[-1]))
                status, _out = run_pip(paths, ["torch", "--index-url", idx], log_cb, abort_event, progress_cb)
                if status == "ok":
                    used_index = idx
                    break
                if status != "nomatch":
                    break           # disk dolu / internet yok: başka sürüm denemek boşa indirme olur
            raise_for_pip(status, paths, need, _out)
            v = verify_engine(paths)
            if not v.get("torch"):
                raise EngineBrokenError(v.get("err", "torch import failed"))

        log_cb(msg("log_engine_libs"))
        status, _out = run_pip(paths, ["--upgrade"] + ENGINE_PIP, log_cb, abort_event, progress_cb)
        raise_for_pip(status, paths, 2.0, _out)
        v = verify_engine(paths)
        if not v.get("torch"):
            raise EngineBrokenError(v.get("err", "torch import failed"))
        if kind == "cuda" and not v.get("cuda_build"):
            # Bir paket PyTorch'u CUDA'sız sürümle değiştirdiyse doğru sürümü geri kur.
            remove_torch(paths)
            status, _out = "nomatch", ""
            for idx in ([used_index] if used_index else torch_index_urls(hw)):
                status, _out = run_pip(paths, ["torch", "--index-url", idx], log_cb, abort_event, progress_cb)
                if status != "nomatch":
                    break
            raise_for_pip(status, paths, need, _out)
            v = verify_engine(paths)
        if not v.get("pipes"):
            status, _out = run_pip(paths, ["--upgrade", "diffusers", "transformers", "accelerate"],
                                   log_cb, abort_event, progress_cb)
            raise_for_pip(status, paths, 1.0, _out)
            v = verify_engine(paths)
            if not v.get("pipes"):
                raise RuntimeError("diffusers: " + str(v.get("derr", "required pipelines missing")))
        info = {"torch": v.get("torch"), "cuda_build": v.get("cuda_build"), "cuda_ok": bool(v.get("cuda_ok")),
                "gpu": v.get("gpu", ""), "diffusers": v.get("diffusers"),
                "python": "%d.%d" % sys.version_info[:2], "date": time.strftime("%Y-%m-%d %H:%M")}
        with open(paths.engine_marker, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        return info
    finally:
        clean_dir(paths.tmp)


# =============================================================================
# Veri klasörünü taşıma ve eski sürüm kalıntıları
# =============================================================================
MOVE_SKIP = ("tmp", "cache", "ollama", "ollama_models")


def data_move_size(old_root):
    if not os.path.isdir(old_root):
        return 0
    return sum(folder_size_bytes(os.path.join(old_root, n)) for n in os.listdir(old_root) if n not in MOVE_SKIP)


def _copy_file_chunked(src, dst, progress, chunk=8 * MB):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(chunk)
            if not b:
                break
            fo.write(b)
            progress(len(b))
    shutil.copystat(src, dst)


def move_data_root(old_root, new_root, progress_cb=None, log_cb=None):
    """
    Veri klasörünü taşır (Ollama ve önbellekler hariç). Hata olursa yarım kopya silinir,
    eski klasör aynen kalır.
    """
    old_root, new_root = os.path.abspath(old_root), os.path.abspath(new_root)
    if os.path.normcase(old_root) == os.path.normcase(new_root):
        return new_root
    if os.path.normcase(new_root).startswith(os.path.normcase(old_root) + os.sep):
        raise RuntimeError("new folder is inside the old one")
    if not is_writable_dir(new_root):
        raise RuntimeError("new folder is not writable")
    items = [n for n in os.listdir(old_root) if n not in MOVE_SKIP] if os.path.isdir(old_root) else []
    clash = [n for n in items if os.path.exists(os.path.join(new_root, n))]
    if clash:
        raise RuntimeError("target already contains: " + ", ".join(clash))
    total = sum(folder_size_bytes(os.path.join(old_root, n)) for n in items)
    same_dev = False
    try:
        same_dev = os.stat(old_root).st_dev == os.stat(new_root).st_dev
    except OSError:
        pass
    if same_dev:
        for n in items:
            os.replace(os.path.join(old_root, n), os.path.join(new_root, n))
        if progress_cb:
            progress_cb(total, total)
    else:
        check_space(new_root, total / GB + 1.0)
        done = [0]

        def prog(n):
            done[0] += n
            if progress_cb:
                progress_cb(done[0], total)

        copied = []
        try:
            for n in items:
                src = os.path.join(old_root, n)
                dst = os.path.join(new_root, n)
                copied.append(dst)
                if os.path.isdir(src):
                    for r, _dirs, files in os.walk(src):
                        for f in files:
                            s = os.path.join(r, f)
                            _copy_file_chunked(s, os.path.join(dst, os.path.relpath(s, src)), prog)
                        for d in _dirs:
                            os.makedirs(os.path.join(dst, os.path.relpath(os.path.join(r, d), src)), exist_ok=True)
                else:
                    _copy_file_chunked(src, dst, prog)
        except BaseException:
            for p in copied:
                delete_path(p)
            raise
        for n in items:
            delete_path(os.path.join(old_root, n))
    for n in ("tmp", "cache"):
        delete_path(os.path.join(old_root, n))
    try:
        if not os.listdir(old_root):
            os.rmdir(old_root)
    except OSError:
        pass
    try:
        fix_venv_home(new_root, old_root)      # programa özel Python da taşındıysa ortam onu yeni yerinde bulsun
    except Exception:
        pass
    return new_root


def legacy_user_site_torch():
    """
    v1.0.0 PyTorch'u genel Python'un kullanıcı klasörüne kuruyordu ve yer bitince bozuk kalabiliyordu.
    Bozuk (içe aktarılamayan) bir kurulum + pip önbelleği bulunursa boyutlarıyla bildirir.
    """
    try:
        import site
        us = site.getusersitepackages()
    except Exception:
        return None
    torch_dir = os.path.join(us, "torch")
    if not os.path.isdir(torch_dir):
        return None
    py = base_python() or sys.executable
    try:
        r = subprocess.run([py, "-c", "import torch"], capture_output=True, text=True, timeout=300,
                           env=sub_env(drop_ours=True), **no_window_kwargs())
        if r.returncode == 0:
            return None          # çalışan bir torch: başka bir program kullanıyor olabilir, dokunma
    except Exception:
        return None
    dirs = [torch_dir] + [p for p in glob.glob(os.path.join(us, "torch-*.dist-info"))] + \
           [os.path.join(us, n) for n in ("torchgen", "functorch") if os.path.isdir(os.path.join(us, n))]
    if IS_WIN:
        pip_cache = os.path.join(os.environ.get("LOCALAPPDATA", ""), "pip", "cache")
    else:
        pip_cache = os.path.join(os.path.expanduser("~"), ".cache", "pip")
    size = sum(folder_size_bytes(d) for d in dirs)
    cache_size = folder_size_bytes(pip_cache) if os.path.isdir(pip_cache) else 0
    return {"dirs": dirs, "pip_cache": pip_cache if cache_size else None, "size_gb": (size + cache_size) / GB,
            "drive": drive_label(us)}


def legacy_cleanup(info):
    before = sum(folder_size_bytes(d) for d in info["dirs"])
    for d in info["dirs"]:
        delete_path(d)
    if info.get("pip_cache"):
        before += folder_size_bytes(info["pip_cache"])
        clean_dir(info["pip_cache"])
    return before / GB


# =============================================================================
# Ollama — yerel ve ağdaki sunucular
# =============================================================================
OLLAMA_PORT = 11434
LOCAL_HOST = f"127.0.0.1:{OLLAMA_PORT}"
OLLAMA_WIN_SETUP = "https://ollama.com/download/OllamaSetup.exe"
_PRIVATE_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]


def is_private_lan_ip(ip):
    """Yalnızca ev/ofis ağı adresleri (internetteki adreslere asla bağlanılmaz)."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if a.version != 4 or not any(a in n for n in _PRIVATE_NETS):
        return False
    last = int(str(a).rsplit(".", 1)[1])
    return last not in (0, 255)


def normalize_host(text):
    """'192.168.1.50', 'http://192.168.1.50:11434/' → '192.168.1.50:11434'; geçersizse None."""
    s = re.sub(r"^https?://", "", (text or "").strip(), flags=re.I).strip("/ ")
    if not s:
        return None
    host, _, port = s.partition(":")
    port = port or str(OLLAMA_PORT)
    if not port.isdigit() or not (0 < int(port) < 65536):
        return None
    if host in ("localhost", "127.0.0.1"):
        return f"127.0.0.1:{int(port)}"
    if not is_private_lan_ip(host):
        return None
    return f"{host}:{int(port)}"


def model_in_list(name, names):
    return name in names or (":" not in name and f"{name}:latest" in names)


class OllamaClient:
    """
    Tek bir Ollama sunucusuyla konuşur. Tüm istekler TEK kalıcı bağlantı havuzundan geçer
    (her istekte yeni bağlantı açılmaz → port tükenmesi olmaz). Sunucuya ulaşılamazsa yeni
    denemeler giderek uzayan aralıklarla yapılır (2 → 4 → 8 … en fazla 60 sn).
    """

    def __init__(self, host=LOCAL_HOST):
        self.host = host
        self.base = "http://" + host
        self.is_local = host.split(":")[0] in ("127.0.0.1", "localhost")
        self._session = None
        self._lock = threading.Lock()
        self.fails = 0
        self.next_try = 0.0

    def _sess(self):
        with self._lock:
            if self._session is None:
                import requests
                from requests.adapters import HTTPAdapter
                s = requests.Session()
                s.trust_env = False          # yerel ağ isteklerini sistem proxy'sine gönderme
                s.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=0))
                self._session = s
            return self._session

    def close(self):
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None

    def reset_backoff(self):
        self.fails, self.next_try = 0, 0.0

    def in_backoff(self):
        return time.time() < self.next_try

    def _call(self, method, path, timeout, force=False, **kw):
        if not force and self.in_backoff():
            raise NetworkError(f"{self.host} unreachable (waiting before retry)")
        import requests
        try:
            r = self._sess().request(method, self.base + path, timeout=timeout, **kw)
        except requests.exceptions.RequestException as e:
            self.fails += 1
            self.next_try = time.time() + min(60.0, 2.0 ** min(self.fails, 6))
            raise NetworkError(f"{self.host}: {type(e).__name__}") from e
        self.fails, self.next_try = 0, 0.0
        return r

    def alive(self, timeout=2.5, force=False):
        try:
            return self._call("GET", "/api/version", timeout, force=force).status_code == 200
        except NetworkError:
            return False

    def version(self):
        try:
            return self._call("GET", "/api/version", 5).json().get("version", "?")
        except Exception:
            return "?"

    def installed_models(self):
        try:
            r = self._call("GET", "/api/tags", 10)
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    def pull(self, name, progress_cb, abort_event):
        r = self._call("POST", "/api/pull", (10, 600), json={"model": name, "stream": True}, stream=True)
        with r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if abort_event is not None and abort_event.is_set():
                    raise AbortedError("user")
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                if d.get("error"):
                    err = str(d["error"])
                    if "no space" in err.lower() or "not enough space" in err.lower():
                        raise DiskFullError("Ollama", 0.0, 0.0)
                    raise RuntimeError(err)
                if progress_cb and d.get("total"):
                    progress_cb(d.get("completed", 0), d["total"])
        return True

    def delete(self, name):
        self._call("DELETE", "/api/delete", 30, json={"model": name})

    def chat_json(self, model, system, user, schema, timeout=(10, 900), temperature=0.4, max_tokens=None):
        options = {"temperature": temperature, "num_ctx": 4096}
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        payload = {
            "model": model, "stream": False, "format": schema, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": options,
        }
        if self.is_local:
            payload["keep_alive"] = 0      # yerelde işi bitince VRAM'i hemen boşalt
        r = self._call("POST", "/api/chat", timeout, json=payload)
        if r.status_code == 400 and "think" in r.text.lower():      # eski Ollama 'think' tanımaz
            payload.pop("think", None)
            r = self._call("POST", "/api/chat", timeout, json=payload)
        if r.status_code == 400 and "format" in r.text.lower():     # JSON şeması desteklenmiyorsa
            payload["format"] = "json"
            r = self._call("POST", "/api/chat", timeout, json=payload)
        r.raise_for_status()
        return parse_json_loose(r.json().get("message", {}).get("content", ""))

    def loaded_models(self):
        try:
            return [m.get("name", "") for m in self._call("GET", "/api/ps", 5).json().get("models", [])]
        except Exception:
            return []

    def unload_all(self, wait_sec=15):
        """Yüklü LLM'leri VRAM'den boşaltır. SADECE yerel sunucuda kullanılır (ağdakilere dokunulmaz)."""
        if not self.is_local:
            return []
        names = self.loaded_models()
        for name in names:
            try:
                self._call("POST", "/api/generate", 15, json={"model": name, "keep_alive": 0})
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < wait_sec and self.loaded_models():
            time.sleep(0.5)
        return names


# --- yerel Ollama: kurulum yeri, kendi ayarları, başlatma ---
def _localappdata():
    return os.environ.get("LOCALAPPDATA", "")


def ollama_app_exe(paths=None):
    """Ollama masaüstü uygulaması (ayarlarını sunucuya o uygular)."""
    if not IS_WIN:
        return None
    cands = [os.path.join(_localappdata(), "Programs", "Ollama", "ollama app.exe")]
    if paths is not None:
        cands.append(os.path.join(paths.ollama_dir, "ollama app.exe"))
    cli = shutil.which("ollama")
    if cli:
        cands.append(os.path.join(os.path.dirname(cli), "ollama app.exe"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def ollama_cli_exe(paths=None):
    cands = [shutil.which("ollama")]
    if IS_WIN:
        cands.append(os.path.join(_localappdata(), "Programs", "Ollama", "ollama.exe"))
        if paths is not None:
            cands.append(os.path.join(paths.ollama_dir, "ollama.exe"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def ollama_installed(paths=None):
    return bool(ollama_app_exe(paths) or ollama_cli_exe(paths))


def read_ollama_app_settings():
    """Ollama uygulamasının kendi ayarları (%LOCALAPPDATA%\\Ollama\\db.sqlite) — salt okunur."""
    if not IS_WIN:
        return {}
    db = os.path.join(_localappdata(), "Ollama", "db.sqlite")
    if not os.path.exists(db):
        return {}
    import sqlite3
    from pathlib import Path
    uri = Path(db).as_uri()
    for suffix in ("?mode=ro", "?immutable=1"):
        try:
            con = sqlite3.connect(uri + suffix, uri=True, timeout=2)
            try:
                try:
                    row = con.execute("SELECT models, expose FROM settings WHERE id = 1").fetchone()
                    return {"models": (row[0] or "") if row else "", "expose": bool(row[1]) if row else False}
                except sqlite3.OperationalError:
                    row = con.execute("SELECT models FROM settings WHERE id = 1").fetchone()
                    return {"models": (row[0] or "") if row else "", "expose": False}
            finally:
                con.close()
        except Exception:
            continue
    return {}


def user_env_var(name):
    """Kalıcı kullanıcı ortam değişkeni (kayıt defterindeki güncel değer, yoksa süreç ortamı)."""
    if IS_WIN:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val, _typ = winreg.QueryValueEx(k, name)
                if val:
                    return os.path.expandvars(str(val))
        except Exception:
            pass
    return os.environ.get(name, "")


def set_user_env_var(name, value):
    """Kalıcı kullanıcı ortam değişkeni yazar (sadece kullanıcı onayıyla çağrılır)."""
    os.environ[name] = value
    if not IS_WIN:
        return False
    try:
        import winreg
        import ctypes
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
        res = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, ctypes.c_wchar_p("Environment"),
                                                 0x0002, 5000, ctypes.byref(res))
        return True
    except Exception:
        return False


def ollama_models_dir():
    """
    Yerel Ollama'nın modelleri indirdiği klasör ve kaynağı ('app' | 'env' | 'default').
    Ollama uygulaması, kendi 'Model location' ayarı doluysa onu, değilse OLLAMA_MODELS'i kullanır.
    """
    s = read_ollama_app_settings()
    if s.get("models") and os.path.isdir(s["models"]):
        return s["models"], "app"
    env = user_env_var("OLLAMA_MODELS")
    if env:
        return env, "env"
    return os.path.join(os.path.expanduser("~"), ".ollama", "models"), "default"


def installed_models_fs(models_dir):
    """Ollama çalışmıyorken bile hangi modellerin indiği (manifests klasöründen, ağ isteği yok)."""
    lib = os.path.join(models_dir, "manifests", "registry.ollama.ai", "library")
    out = []
    if os.path.isdir(lib):
        for model in os.listdir(lib):
            mdir = os.path.join(lib, model)
            if os.path.isdir(mdir):
                for tag in os.listdir(mdir):
                    if os.path.isfile(os.path.join(mdir, tag)):
                        out.append(f"{model}:{tag}")
    return out


def start_local_ollama(paths=None, client=None, wait_sec=45):
    """
    Yerel Ollama'yı başlatır. Masaüstü uygulaması varsa ONU başlatır (pencere açmadan), çünkü
    kullanıcının 'Model location' gibi ayarlarını sunucuya uygulama aktarır. Doğrudan 'ollama serve'
    sadece uygulama yoksa kullanılır; o durumda da uygulamanın model konumu ayarı aktarılır.
    """
    client = client or OllamaClient(LOCAL_HOST)
    if client.alive(timeout=1.5, force=True):
        return True
    env = os.environ.copy()
    app = ollama_app_exe(paths)
    try:
        if app:
            subprocess.Popen([app, "hidden"], env=env, cwd=os.path.dirname(app), **no_window_kwargs())
        else:
            cli = ollama_cli_exe(paths)
            if not cli:
                return False
            s = read_ollama_app_settings()
            if s.get("models") and os.path.isdir(s["models"]):
                env["OLLAMA_MODELS"] = s["models"]
            subprocess.Popen([cli, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             **no_window_kwargs())
    except Exception:
        return False
    t0 = time.time()
    while time.time() - t0 < wait_sec:
        if client.alive(timeout=1.5, force=True):
            client.reset_backoff()
            return True
        time.sleep(1.0)
    return False


def stop_local_ollama():
    if not IS_WIN:
        return
    for im in ("ollama app.exe", "ollama.exe"):
        try:
            subprocess.run(["taskkill", "/f", "/im", im], capture_output=True, timeout=15, **no_window_kwargs())
        except Exception:
            pass
    time.sleep(2.0)


def restart_local_ollama(paths=None, client=None):
    stop_local_ollama()
    return start_local_ollama(paths, client)


def install_ollama_windows(paths, log_cb, progress_cb, abort_event):
    """Ollama'yı veri klasörüne kurar; LLM modelleri de veri klasörüne iner (kullanıcı onayıyla)."""
    check_space(paths.root, 5.0)
    setup = os.path.join(paths.tmp, "OllamaSetup.exe")
    download_file(OLLAMA_WIN_SETUP, setup, progress_cb, abort_event, timeout=60, need_gb=4.0)
    os.makedirs(paths.ollama_models, exist_ok=True)
    set_user_env_var("OLLAMA_MODELS", paths.ollama_models)
    if progress_cb:
        progress_cb(-1, 0)
    proc = subprocess.Popen([setup, "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/DIR=" + paths.ollama_dir])
    while proc.poll() is None:
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
        time.sleep(1.0)
    try:
        os.remove(setup)
    except OSError:
        pass
    return start_local_ollama(paths, wait_sec=90)


# --- ağdaki Ollama sunucuları: ağı yormadan bulma ---
_MAC_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2}\b")
_IP_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")


def local_ipv4s():
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    try:
        # UDP 'connect' paket göndermez; sadece hangi arayüzün kullanılacağını öğrenir.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


def _arp_table_text():
    try:
        if IS_WIN:
            out = subprocess.run(["arp", "-a"], capture_output=True, timeout=5, **no_window_kwargs()).stdout
            return out.decode("ascii", "ignore")
        if os.path.exists("/proc/net/arp"):
            with open("/proc/net/arp") as f:
                return f.read()
        out = subprocess.run(["arp", "-an"], capture_output=True, timeout=5).stdout
        return out.decode("ascii", "ignore")
    except Exception:
        return ""


def parse_arp_neighbors(text, own=(), limit=10):
    """ARP tablosundan, gerçek bir cihaz MAC adresi olan özel ağ IP'leri (ağa paket gönderilmez)."""
    ips = []
    for line in (text or "").splitlines():
        mi, mm = _IP_RE.search(line), _MAC_RE.search(line)
        if not mi or not mm:
            continue
        mac = mm.group(0).lower().replace(":", "-")
        if mac in ("ff-ff-ff-ff-ff-ff", "00-00-00-00-00-00") or mac.startswith(("01-00-5e", "33-33")):
            continue
        ip = mi.group(1)
        if is_private_lan_ip(ip) and ip not in own and ip not in ips:
            ips.append(ip)
    return ips[:limit]


def arp_neighbors(limit=10):
    return parse_arp_neighbors(_arp_table_text(), own=local_ipv4s(), limit=limit)


def probe_ollama(host, connect_timeout=0.4, http_timeout=2.0):
    """Tek bir adresi dener: önce kısa bir TCP bağlantısı, port açıksa tek bir HTTP isteği."""
    ip, _, port = host.partition(":")
    try:
        with socket.create_connection((ip, int(port or OLLAMA_PORT)), timeout=connect_timeout):
            pass
    except OSError:
        return None
    try:
        import requests
        with requests.Session() as s:
            s.trust_env = False
            r = s.get(f"http://{host}/api/version", timeout=http_timeout)
            if r.status_code == 200:
                return r.json().get("version", "?")
    except Exception:
        pass
    return None


def discover_lan_ollama(known=(), limit=10, delay=0.3, abort_event=None, probe=probe_ollama, neighbors=None):
    """
    Ağı yormayan arama: 254 adresin tamamı taranmaz. Yalnızca daha önce bilinen sunucular ve
    Windows'un zaten tanıdığı komşu cihazlar (ARP tablosu) SIRAYLA, aralarında beklenerek denenir.
    """
    cands = []
    for h in known:
        h = normalize_host(h)
        if h and not h.startswith("127.") and h not in cands:
            cands.append(h)
    for ip in (neighbors if neighbors is not None else arp_neighbors(limit)):
        h = f"{ip}:{OLLAMA_PORT}"
        if h not in cands:
            cands.append(h)
    cands = cands[:limit + len(known)]
    found = []
    for i, host in enumerate(cands):
        if abort_event is not None and abort_event.is_set():
            break
        if i:
            time.sleep(delay)
        ver = probe(host)
        if ver:
            found.append((host, ver))
    return found


# =============================================================================
# Ağ: görsel sunucusu (bu bilgisayarın modelleri ve LoRA'ları ağdan kullanılır) ve istemcisi
# Sunucu Ollama gibi çalışır: yalnızca kullanıcı açarsa dinler. "Yalnızca bu bilgisayar" (127.0.0.1) ya da
# "ev/ofis ağı" seçilebilir; ağ modunda erişim anahtarı zorunludur. İnternete hiçbir şey açılmaz (yönlendirme
# yapılmaz, yalnızca yerel ağ adresleri kabul edilir). Uyumlu uç noktalar:
#   /api/*            — programın kendi istemcisi (iş kuyruğu, ilerleme, durdurma, LoRA paylaşımı)
#   /sdapi/v1/*       — Automatic1111 uyumlu (SillyTavern, Open WebUI, Krita eklentileri…)
#   /v1/images/*      — OpenAI uyumlu görsel üretimi
# =============================================================================
API_PORT = 7860
API_MAX_BODY = 32 * MB
API_MAX_UPLOAD = 2 * GB
API_MAX_COUNT = 8
API_KEEP_DONE = 60 * 30            # bitmiş işler 30 dk tutulur
NETWORK_LORA_SUBDIR = "network"    # ağdan gelen LoRA'lar: addons\network\


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def new_api_key():
    import secrets
    return secrets.token_urlsafe(18)


def lan_ipv4_addresses():
    """Bu bilgisayarın yerel ağ adresleri (bağlantı kurmadan)."""
    import socket
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if is_private_lan_ip(ip) and ip not in out:
                out.append(ip)
    except Exception:
        pass
    if not out:
        s = None
        try:                                   # UDP "bağlanma" paket göndermez; yalnızca yönlendirme tablosuna bakar
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.168.0.1", 9))
            ip = s.getsockname()[0]
            if is_private_lan_ip(ip):
                out.append(ip)
        except Exception:
            pass
        finally:
            if s is not None:
                s.close()
    return out


def normalize_image_host(text):
    """'192.168.1.20', 'http://192.168.1.20:7860/' → '192.168.1.20:7860'. Yalnızca yerel ağ adresleri."""
    s = re.sub(r"^https?://", "", (text or "").strip(), flags=re.I).strip("/ ")
    if not s:
        return None
    host, _, port = s.partition(":")
    port = port or str(API_PORT)
    if not port.isdigit() or not (0 < int(port) < 65536):
        return None
    if host in ("localhost", "127.0.0.1"):
        return f"127.0.0.1:{int(port)}"
    if not is_private_lan_ip(host):
        return None
    return f"{host}:{int(port)}"


_SHA1_CACHE = {}
_SHA1_LOCK = threading.Lock()


def file_sha1_cached(path, cache_file=None):
    """Dosyanın SHA-1'i; (boyut, değişiklik zamanı) aynıysa önbellekten. İsteğe bağlı diskte saklanır."""
    st = os.stat(path)
    sig = [st.st_size, int(st.st_mtime)]
    key = os.path.normcase(os.path.abspath(path))
    with _SHA1_LOCK:
        if not _SHA1_CACHE and cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, encoding="utf-8") as f:
                    _SHA1_CACHE.update(json.load(f))
            except Exception:
                pass
        hit = _SHA1_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * MB), b""):
            h.update(chunk)
    digest = h.hexdigest()
    with _SHA1_LOCK:
        _SHA1_CACHE[key] = [sig, digest]
        if cache_file:
            try:
                tmp = cache_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(_SHA1_CACHE, f)
                os.replace(tmp, cache_file)
            except Exception:
                pass
    return digest


_A1111_LORA = re.compile(r"<lora:([^:>]+)(?::([-0-9.]+))?(?::[^>]*)?>", re.I)


def parse_a1111_loras(prompt):
    """'a cat <lora:yarn:0.8>' → ('a cat', [('yarn', 0.8)]) — Automatic1111 LoRA yazımı."""
    found = [(m.group(1).strip(), float(m.group(2)) if m.group(2) else 1.0) for m in _A1111_LORA.finditer(prompt or "")]
    clean = re.sub(r"\s{2,}", " ", _A1111_LORA.sub("", prompt or ""))
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r",\s*,+", ",", clean).strip(" ,")
    return clean, found


def snap_size(model, w, h):
    """İstenen boyutu modele uygun hâle getirir: 16'nın katı, 256–2048 arası, alan en çok ~2.4 MP."""
    w, h = int(w or model["base_px"]), int(h or model["base_px"])
    w, h = max(256, min(2048, w)), max(256, min(2048, h))
    area = w * h
    cap = 2.4e6
    if area > cap:
        k = (cap / area) ** 0.5
        w, h = int(w * k), int(h * k)
    return max(256, w // 16 * 16), max(256, h // 16 * 16)


def public_model_info(m):
    """Bir modelin ağa verilecek bilgisi (dosya yolları ve indirme adresleri olmadan)."""
    keep = ("key", "name", "family", "base_px", "neg", "cfg", "fixed_cfg", "samplers", "turbo", "steps", "license",
            "desc", "custom", "size_gb", "vram", "vram_min")
    d = {k: m.get(k) for k in keep if k in m}
    d["family"] = model_family(m)
    for k, v in m.items():
        if k not in d and isinstance(v, (int, float, str, bool, type(None))) and k not in (
                "file", "models_root", "repo", "variant", "misplaced"):
            d[k] = v
    return d


class ApiJob:
    def __init__(self, params, source):
        import uuid
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.source = source
        self.state = "queued"          # queued | running | done | error | stopped
        self.progress = 0.0
        self.text = ""
        self.images = []               # [(png bytes, meta)]
        self.error = None
        self.created = time.time()
        self.finished = None
        self.abort = threading.Event()
        self.done = threading.Event()

    def public(self, since=0):
        import base64
        return {"id": self.id, "state": self.state, "progress": round(self.progress, 1), "text": self.text,
                "error": self.error, "count": len(self.images),
                "images": [{"index": i, "png": base64.b64encode(b).decode("ascii"), "meta": m}
                           for i, (b, m) in enumerate(self.images) if i >= since]}


class ApiServer:
    """
    Basit, bağımlılıksız HTTP sunucusu (Python'un kendi http.server'ı). İşler sırayla, tek tek çalışır;
    gerçek üretimi 'backend' yapar (programın kendi iş sistemi — arayüzdeki üretimle aynı yol).
    """

    def __init__(self, backend, lan=False, port=API_PORT, key=""):
        self.backend = backend
        self.lan = bool(lan)
        self.port = int(port)
        self.key = key or ""
        self.httpd = None
        self.jobs = {}
        self.queue = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.current = None
        self.requests = 0

    # ---- yaşam döngüsü
    def start(self):
        from http.server import ThreadingHTTPServer
        bind = "0.0.0.0" if self.lan else "127.0.0.1"
        handler = _make_api_handler(self)
        self.httpd = ThreadingHTTPServer((bind, self.port), handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def stop(self):
        self.stopping.set()
        self.wake.set()
        with self.lock:
            for j in self.queue:
                j.state, j.error = "stopped", "server stopped"
                j.done.set()
            self.queue = []
            cur = self.current
        if cur is not None:
            cur.abort.set()
            try:
                self.backend.abort(cur)
            except Exception:
                pass
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None

    def urls(self):
        if not self.lan:
            return [f"http://127.0.0.1:{self.port}"]
        return [f"http://{ip}:{self.port}" for ip in lan_ipv4_addresses()] or [f"http://<IP>:{self.port}"]

    # ---- kimlik
    def authorized(self, headers, client_ip):
        if not self.key:
            return True
        if client_ip in ("127.0.0.1", "::1") and not self.lan:
            return True
        import base64
        import hmac
        cands = []
        auth = headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            cands.append(auth[7:].strip())
        elif auth.lower().startswith("basic "):
            try:
                user_pw = base64.b64decode(auth[6:].strip()).decode("utf-8", "replace")
                cands.append(user_pw.partition(":")[2])
            except Exception:
                pass
        if headers.get("X-API-Key"):
            cands.append(headers.get("X-API-Key").strip())
        return any(hmac.compare_digest(c.encode(), self.key.encode()) for c in cands if c)

    def local_request_ok(self, host, origin):
        """Yalnızca bu bilgisayar modunda (anahtarsız) Host ve Origin de bu bilgisayarı göstermeli."""
        if self.lan:
            return True
        loc = ("127.0.0.1", "localhost", "::1")

        def name(v):
            v = (v or "").strip().lower()
            if "://" in v:
                v = v.split("://", 1)[1]
            v = v.split("/", 1)[0]
            if v.startswith("["):
                return v[1:].split("]", 1)[0]
            return v.rsplit(":", 1)[0] if v.count(":") == 1 else v
        if host and name(host) not in loc:
            return False
        if origin and origin.strip().lower() != "null" and name(origin) not in loc:
            return False
        if origin and origin.strip().lower() == "null":
            return False
        return True

    def client_allowed(self, ip):
        """Ağ modunda yalnızca yerel ağdan ve bu bilgisayardan gelen bağlantılar."""
        return ip in ("127.0.0.1", "::1") or is_private_lan_ip(ip)

    # ---- işler
    def submit(self, params, source):
        job = ApiJob(params, source)
        with self.lock:
            self._gc()
            if len(self.queue) >= 20:
                raise ApiError(503, "queue full")
            self.jobs[job.id] = job
            self.queue.append(job)
        self.wake.set()
        return job

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job in self.queue:
                self.queue.remove(job)
                job.state, job.error = "stopped", "cancelled"
                job.done.set()
                return True
        job.abort.set()
        if job is self.current:
            try:
                self.backend.abort(job)
            except Exception:
                pass
        return True

    def _gc(self):
        now = time.time()
        for jid in [k for k, j in self.jobs.items() if j.finished and now - j.finished > API_KEEP_DONE]:
            del self.jobs[jid]

    def _worker(self):
        while not self.stopping.is_set():
            with self.lock:
                job = self.queue.pop(0) if self.queue else None
                self.current = job
            if job is None:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            job.state = "running"
            try:
                self.backend.run(job)
                job.state = "stopped" if job.abort.is_set() and not job.images else "done"
            except AbortedError:
                job.state, job.error = "stopped", "stopped"
            except ApiError as e:
                job.state, job.error = "error", e.message
            except Exception as e:  # noqa
                job.state, job.error = "error", f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                job.progress = 100.0 if job.state == "done" else job.progress
                job.finished = time.time()
                with self.lock:
                    self.current = None
                job.done.set()

    def wait(self, job, timeout=3600):
        job.done.wait(timeout)
        if job.state == "error":
            raise ApiError(500, job.error or "generation failed")
        if job.state == "stopped" and not job.images:
            raise ApiError(409, job.error or "stopped")
        return job

    # ---- istek → parametre dönüştürücüler
    def a1111_params(self, body):
        prompt, loras = parse_a1111_loras(body.get("prompt", ""))
        ov = body.get("override_settings") or {}
        model = ov.get("sd_model_checkpoint") or body.get("model") or self.backend.current_model()
        n = int(body.get("batch_size") or 1) * int(body.get("n_iter") or 1)
        return {"prompt": prompt, "negative": body.get("negative_prompt", ""), "model": model,
                "width": body.get("width"), "height": body.get("height"), "steps": body.get("steps"),
                "cfg": body.get("cfg_scale"), "seed": body.get("seed", -1), "count": n,
                "loras": [{"name": a, "weight": w} for a, w in loras], "raw_prompt": False}

    def openai_params(self, body):
        size = str(body.get("size") or "")
        w = h = None
        if "x" in size:
            try:
                w, h = (int(x) for x in size.lower().split("x", 1))
            except ValueError:
                w = h = None
        prompt, loras = parse_a1111_loras(body.get("prompt", ""))
        return {"prompt": prompt, "negative": "", "model": body.get("model") or self.backend.current_model(),
                "width": w, "height": h, "count": int(body.get("n") or 1), "seed": -1,
                "loras": [{"name": a, "weight": wt} for a, wt in loras], "raw_prompt": False}


def _make_api_handler(server):
    from http.server import BaseHTTPRequestHandler
    import base64
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AILocalImageStudio"

        def log_message(self, *a):          # konsola yazma
            pass

        # -- yardımcılar
        def _send(self, status, obj=None, raw=None, ctype="application/json; charset=utf-8"):
            body = raw if raw is not None else json.dumps(obj if obj is not None else {}, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, msg):
            self._send(status, {"error": msg, "detail": msg})

        def _body(self, limit=API_MAX_BODY):
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise ApiError(413, "request too large")
            data = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(data.decode("utf-8") or "{}")
            except Exception:
                raise ApiError(400, "invalid JSON")

        def _guard(self):
            ip = self.client_address[0]
            if not server.client_allowed(ip):
                self._err(403, "only local network clients are allowed")
                return False
            if not server.local_request_ok(self.headers.get("Host"), self.headers.get("Origin")):
                # anahtarsız (yalnızca bu bilgisayar) modda bir web sayfasının tarayıcı üzerinden istek atmasını
                # (DNS rebinding / siteler arası istek) engeller
                self._err(403, "requests from web pages are not allowed in 'this computer only' mode")
                return False
            if not server.authorized(self.headers, ip):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="AI Local Image Studio"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
            server.requests += 1
            return True

        def _route(self, method):
            u = urlparse(self.path)
            path, qs = u.path.rstrip("/") or "/", parse_qs(u.query)
            if path == "/api/ping":
                return self._send(200, {"app": "AI Local Image Studio", "version": server.backend.version()})
            if not self._guard():
                return
            b = server.backend
            try:
                if method == "GET" and path == "/api/info":
                    return self._send(200, b.info(server))
                if method == "GET" and path == "/api/loras":
                    return self._send(200, {"loras": b.loras()})
                if method == "GET" and path == "/api/loras/file":
                    sha = (qs.get("sha1") or [""])[0]
                    p = b.lora_path(sha)
                    if not p:
                        raise ApiError(404, "LoRA not found")
                    return self._send_file(p)
                if method == "POST" and path == "/api/loras/upload":
                    return self._upload(qs)
                if method == "POST" and path == "/api/jobs":
                    params = b.prepare(self._body(), source=self.client_address[0], kind="ails")
                    job = server.submit(params, self.client_address[0])
                    return self._send(200, {"id": job.id})
                if path.startswith("/api/jobs/"):
                    jid = path.rsplit("/", 1)[1]
                    job = server.jobs.get(jid)
                    if job is None:
                        raise ApiError(404, "job not found")
                    if method == "DELETE":
                        server.cancel(jid)
                        return self._send(200, {"ok": True})
                    since = int((qs.get("since") or ["0"])[0] or 0)
                    return self._send(200, job.public(since))
                if method == "POST" and path == "/api/generate":
                    params = b.prepare(self._body(), source=self.client_address[0], kind="ails")
                    job = server.wait(server.submit(params, self.client_address[0]))
                    return self._send(200, job.public(0))
                # ---- Automatic1111 uyumlu
                if path.startswith("/sdapi/v1/"):
                    return self._a1111(method, path[len("/sdapi/v1/"):])
                # ---- OpenAI uyumlu
                if method == "GET" and path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [
                        {"id": m["key"], "object": "model", "owned_by": "local", "name": m["name"]}
                        for m in b.models()]})
                if method == "POST" and path == "/v1/images/generations":
                    body = self._body()
                    params = b.prepare(server.openai_params(body), source=self.client_address[0], kind="openai")
                    job = server.wait(server.submit(params, self.client_address[0]))
                    data = [{"b64_json": base64.b64encode(png).decode("ascii"),
                             "revised_prompt": meta.get("prompt", "")} for png, meta in job.images]
                    return self._send(200, {"created": int(time.time()), "data": data})
                raise ApiError(404, "unknown endpoint")
            except ApiError as e:
                return self._err(e.status, e.message)
            except Exception as e:  # noqa
                return self._err(500, f"{type(e).__name__}: {str(e)[:200]}")

        def _a1111(self, method, ep):
            b = server.backend
            if method == "GET" and ep == "sd-models":
                return self._send(200, [{"title": m["name"], "model_name": m["key"], "hash": None, "sha256": None,
                                         "filename": m["key"], "config": None} for m in b.models()])
            if method == "GET" and ep == "loras":
                return self._send(200, [{"name": x["name"], "alias": x["name"], "path": x["name"],
                                         "metadata": {"base": x["base"]}} for x in b.loras()])
            if method == "GET" and ep == "options":
                return self._send(200, {"sd_model_checkpoint": b.current_model(), "samples_format": "png"})
            if method == "POST" and ep == "options":
                body = self._body()
                if body.get("sd_model_checkpoint"):
                    b.set_default_model(str(body["sd_model_checkpoint"]))
                return self._send(200, {})
            if method == "GET" and ep in ("samplers", "schedulers"):
                return self._send(200, [{"name": "Euler a", "aliases": []}, {"name": "DPM++ 2M Karras", "aliases": []}])
            if method == "GET" and ep in ("upscalers", "sd-vae", "hypernetworks", "embeddings", "prompt-styles",
                                          "face-restorers", "realesrgan-models", "scripts", "extensions"):
                return self._send(200, {"loaded": {}, "skipped": {}} if ep == "embeddings" else [])
            if method == "GET" and ep == "progress":
                cur = server.current
                return self._send(200, {"progress": (cur.progress / 100.0) if cur else 0.0, "eta_relative": 0,
                                        "state": {"job_count": len(server.queue) + (1 if cur else 0)},
                                        "current_image": None, "textinfo": cur.text if cur else ""})
            if method == "POST" and ep == "interrupt":
                if server.current is not None:
                    server.cancel(server.current.id)
                return self._send(200, {})
            if method == "POST" and ep in ("refresh-checkpoints", "refresh-loras"):
                return self._send(200, {})
            if method == "POST" and ep == "txt2img":
                body = self._body()
                params = b.prepare(server.a1111_params(body), source=self.client_address[0], kind="a1111")
                job = server.wait(server.submit(params, self.client_address[0]))
                imgs = [base64.b64encode(png).decode("ascii") for png, _m in job.images]
                info = job.images[0][1] if job.images else {}
                return self._send(200, {"images": imgs, "parameters": body, "info": json.dumps(info, ensure_ascii=False)})
            raise ApiError(404, "unsupported A1111 endpoint: " + ep)

        def _send_file(self, p):
            size = os.path.getsize(p)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            from urllib.parse import quote
            fn = os.path.basename(p)
            ascii_fn = fn.encode("ascii", "replace").decode("ascii").replace('"', "").replace("?", "_")
            self.send_header("Content-Disposition",       # başlıklar latin-1: Türkçe ad UTF-8 olarak ayrıca verilir
                             "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_fn, quote(fn)))
            self.end_headers()
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 * MB)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _upload(self, qs):
            b = server.backend
            if not b.allow_upload():
                raise ApiError(403, "this server does not accept LoRA uploads")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > API_MAX_UPLOAD:
                raise ApiError(413, "LoRA file too large")
            name = safe_folder_name((qs.get("name") or ["lora"])[0], "lora")[:60]
            want = ((qs.get("sha1") or [""])[0] or "").lower()
            tmp = b.upload_tmp_path()
            h = hashlib.sha1()
            left = n
            with open(tmp, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 * MB, left))
                    if not chunk:
                        break
                    h.update(chunk)
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                os.remove(tmp)
                raise ApiError(400, "upload incomplete")
            if want and h.hexdigest() != want:
                os.remove(tmp)
                raise ApiError(400, "checksum mismatch")
            return self._send(200, b.store_upload(tmp, name, h.hexdigest()))

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


class ImageServerClient:
    """Ağdaki bir AI Local Image Studio sunucusuyla konuşur (tek kalıcı bağlantı havuzu, sistem proxy'si yok)."""

    def __init__(self, host, key=""):
        self.host = host
        self.base = "http://" + host
        self.key = key or ""
        self._s = None

    def _sess(self):
        if self._s is None:
            import requests
            from requests.adapters import HTTPAdapter
            s = requests.Session()
            s.trust_env = False
            s.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=0))
            if self.key:
                s.headers["Authorization"] = "Bearer " + self.key
            self._s = s
        return self._s

    def close(self):
        if self._s is not None:
            try:
                self._s.close()
            except Exception:
                pass
            self._s = None

    def _req(self, method, path, timeout=15, **kw):
        import requests
        try:
            r = self._sess().request(method, self.base + path, timeout=timeout, **kw)
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"{self.host}: {type(e).__name__}") from e
        if r.status_code == 401:
            raise ApiError(401, "wrong or missing access key")
        if r.status_code >= 400:
            try:
                msg = r.json().get("error") or r.text[:200]
            except Exception:
                msg = r.text[:200]
            raise ApiError(r.status_code, msg)
        return r

    def ping(self, timeout=3):
        try:
            return self._req("GET", "/api/ping", timeout=timeout).json()
        except Exception:
            return None

    def info(self):
        return self._req("GET", "/api/info", timeout=10).json()

    def loras(self):
        return self._req("GET", "/api/loras", timeout=120).json().get("loras", [])

    def upload_lora(self, path, sha1, name, progress_cb=None, abort_event=None):
        size = os.path.getsize(path)

        def gen():
            sent = 0
            with open(path, "rb") as f:
                while True:
                    if abort_event is not None and abort_event.is_set():
                        raise AbortedError("user")
                    chunk = f.read(1 * MB)
                    if not chunk:
                        break
                    sent += len(chunk)
                    if progress_cb:
                        progress_cb(sent, size)
                    yield chunk
        from urllib.parse import quote
        return self._req("POST", f"/api/loras/upload?name={quote(name)}&sha1={sha1}", timeout=(10, 600),
                         data=gen(), headers={"Content-Length": str(size),
                                              "Content-Type": "application/octet-stream"}).json()

    def download_lora(self, sha1, dest, progress_cb=None, abort_event=None):
        r = self._req("GET", f"/api/loras/file?sha1={sha1}", timeout=(10, 600), stream=True)
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        tmp = dest + ".part"
        with r, open(tmp, "wb") as f:
            for chunk in r.iter_content(1 * MB):
                if abort_event is not None and abort_event.is_set():
                    raise AbortedError("user")
                f.write(chunk)
                got += len(chunk)
                if progress_cb:
                    progress_cb(got, total)
        os.replace(tmp, dest)
        return dest

    def submit(self, params):
        return self._req("POST", "/api/jobs", timeout=30, json=params).json()["id"]

    def job(self, jid, since=0):
        return self._req("GET", f"/api/jobs/{jid}?since={since}", timeout=60).json()

    def cancel(self, jid):
        try:
            self._req("DELETE", f"/api/jobs/{jid}", timeout=10)
        except Exception:
            pass


# =============================================================================
# LLM isteği (prompt hazırlama)
# =============================================================================
def parse_json_loose(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("LLM did not return valid JSON")


def llm_word_limit(model, user_text):
    """Kısa istek → kısa prompt: kullanıcının kelime sayısına göre en fazla kaç kelime yazılabilir."""
    n = max(1, len((user_text or "").split()))
    if model["prompt_style"] == "tags":
        return min(60, max(20, 6 + 4 * n))
    return min(90, max(25, 10 + 5 * n))


def build_llm_request(model, user_text, extra_text, auto_fields):
    """
    LLM için sistem talimatı, kullanıcı mesajı ve JSON şeması üretir.
    auto_fields: LLM'in seçeceği ayarlar kümesi {"aspect","orientation","creativity"}.
    Kural: kullanıcının yazdığı anlam aynen korunur, uydurma yok, kısa istek kısa prompt.
    """
    limit = llm_word_limit(model, user_text)
    n_words = max(1, len((user_text or "").split()))
    if model["prompt_style"] == "tags":
        style_rule = ("Write it as comma-separated tags and short phrases, main subject first. "
                      f"At most {limit} words.")
    else:
        style_rule = f"Write it as one or two plain English sentences, main subject first. At most {limit} words."
    if model["neg"]:
        neg_rule = ("an English comma-separated list of things to avoid: general defects (blurry, low quality, "
                    "deformed, watermark, text) plus everything the user wants to avoid. Maximum 25 words.")
    else:
        neg_rule = ("an empty string, because this model ignores negative prompts. If the user wants to avoid "
                    "something important, express it positively inside the positive prompt instead.")
    lines = [
        "You turn a short image description into an English prompt for an image model.",
        "The description can be in any language. Return ONLY a JSON object.",
        "Field \"positive\": the English prompt. " + style_rule,
        "RULE 1 - Be faithful: translate exactly what the user wrote. Keep every subject, count, color, clothing, "
        "age, gender, nationality, ethnicity and place with the same meaning. Never swap a word for a different one.",
        "RULE 2 - Invent nothing: do not add people, objects, places, nationality, ethnicity, actions, story, mood, "
        "clothing, colors or background details that the user did not mention. Do not assume any country or "
        "culture the user did not name.",
        f"RULE 3 - Short in, short out: the user wrote {n_words} word(s). Besides the translation you may only "
        "add a few general quality words (for example: photo, natural light, sharp focus, highly detailed).",
        "If a person is in the image you may add 'detailed face'.",
        "Field \"negative\": " + neg_rule,
        "If the user gave extra instructions, follow them.",
    ]
    props = {"positive": {"type": "string"}, "negative": {"type": "string"}}
    required = ["positive", "negative"]
    if "aspect" in auto_fields:
        props["aspect"] = {"type": "string", "enum": list(ASPECTS.keys())}
        required.append("aspect")
        lines.append("Field \"aspect\": best aspect ratio for this image (landscapes and wide scenes 16:9 or 21:9, "
                     "single subjects 1:1 or 4:3 or 3:2).")
    if "orientation" in auto_fields:
        props["orientation"] = {"type": "string", "enum": ["landscape", "portrait"]}
        required.append("orientation")
        lines.append("Field \"orientation\": \"portrait\" for standing people, phone wallpapers, tall subjects; "
                     "otherwise \"landscape\".")
    if "creativity" in auto_fields:
        props["creativity"] = {"type": "integer", "minimum": 1, "maximum": 10}
        required.append("creativity")
        lines.append("Field \"creativity\": 1-10. Low (2-4) when the user gave precise details, high (6-8) when the "
                     "request is short, abstract or artistic.")
    schema = {"type": "object", "properties": props, "required": required}
    user = "Image description:\n" + user_text.strip()
    must = requested_demonyms(user_text)
    if must:
        user += "\n\nThe description says: " + ", ".join(must) + ". Use exactly this word in the prompt."
    if extra_text.strip():
        user += "\n\nExtra instructions / things to avoid:\n" + extra_text.strip()
    return "\n".join(lines), user, schema


def llm_too_long(model, user_text, positive):
    """LLM kısa isteğe uzun prompt yazdıysa (kendinden ayrıntı eklemiş olması çok olası) True."""
    return len((positive or "").split()) > int(llm_word_limit(model, user_text) * 1.3)


def build_llm_cleanup(model, user_text, positive):
    """İkinci geçiş: prompt'tan kullanıcının söylemediği her şeyi sildirir."""
    limit = llm_word_limit(model, user_text)
    system = "\n".join([
        "You clean up an English image prompt so that it matches the user's description exactly.",
        "Return ONLY a JSON object with the field \"positive\".",
        "Delete every detail the description does not mention: places, buildings, objects, clothing, colors, "
        "actions, poses, mood, time of day, era, culture, nationality, ethnicity.",
        "Keep the English translation of everything the description says, with the same meaning.",
        "You may keep a few general quality words (photo, natural light, sharp focus, highly detailed).",
        f"At most {limit} words. Do not add anything new.",
    ])
    user = "Description:\n" + (user_text or "").strip() + "\n\nPrompt to clean:\n" + (positive or "").strip()
    must = requested_demonyms(user_text)
    if must:
        user += "\n\nThe description says: " + ", ".join(must) + ". Keep exactly this word."
    schema = {"type": "object", "properties": {"positive": {"type": "string"}}, "required": ["positive"]}
    return system, user, schema


def sanitize_llm_result(d, auto_fields):
    out = {"positive": str(d.get("positive", "")).strip(), "negative": str(d.get("negative", "")).strip()}
    if "aspect" in auto_fields:
        a = str(d.get("aspect", "1:1"))
        out["aspect"] = a if a in ASPECTS else "1:1"
    if "orientation" in auto_fields:
        o = str(d.get("orientation", "landscape"))
        out["orientation"] = o if o in ("landscape", "portrait") else "landscape"
    if "creativity" in auto_fields:
        try:
            out["creativity"] = min(10, max(1, int(d.get("creativity", 5))))
        except Exception:
            out["creativity"] = 5
    return out


# --- LLM'in yazdığını kullanıcının isteğine göre denetleme (milliyet/köken, uzunluk) ---
# (İngilizce karşılık, kullanıcı metninde tam kelime kalıbı (tr|en|de), Korece ülke kökleri, LLM çıktısı kalıbı).
# Dil / yazı adları (Türkçe, Arapça, arabic…) ve renkle karışan kelimeler (siyah, beyaz) bilerek yok.
_DEMONYMS = [
    ("Asian", r"asyal[ıi]\w*|asians?|asiat\w*", ["아시아", "동양"],
     r"(?:(?:east|south|southeast|south-east|central|west)[ -])?asians?"),
    ("Turkish", r"türk(ler\w*|ün|e|ü|ten|le|isch\w*|in(nen)?|en)?|turks?|turkish", ["터키", "튀르키예"],
     r"turks?|turkish"),
    ("Japanese", r"japon(lar\w*|un|a|u|dan)?|japanese|japan(isch\w*|erin(nen)?|er|ern)", ["일본"], r"japanese"),
    ("Korean", r"koreli\w*|koreans?|korean(isch\w*|erin(nen)?|er|ern)", ["한국"], r"koreans?"),
    ("Chinese", r"çinli\w*|chinese|chines(isch\w*|in(nen)?|en?)", ["중국"], r"chinese"),
    ("Indian", r"hintli\w*|indians?|ind(isch\w*|erin(nen)?|er|ern)", ["인도"], r"indians?"),
    ("African", r"afrikal[ıi]\w*|africans?|afrikan(isch\w*|erin(nen)?|er|ern)", ["아프리카"], r"africans?"),
    ("European", r"avrupal[ıi]\w*|europeans?|europä(isch\w*|erin(nen)?|er|ern)", ["유럽"], r"europeans?"),
    ("Arab", r"arap(lar\w*)?|arabs?|arabian|arab(erin(nen)?|er|ern)", ["아랍"], r"arabs?|arabian"),
    ("Middle Eastern", r"ortadoğulu\w*|nahöstlich\w*", ["중동"], r"middle[ -]eastern"),
    ("Latin American", r"latin(ler\w*|li\w*)|latin[ao]s?|hispanics?|lateinamerikan\w*", ["라틴"],
     r"latin[ao]s?|latinx|hispanics?|latin[ -]american"),
    ("Russian", r"rus(lar\w*|un|a)?|russians?|russ(isch\w*|in(nen)?|en?)", ["러시아"], r"russians?"),
    ("American", r"amerikal[ıi]\w*|americans?|amerikan(isch\w*|erin(nen)?|er|ern)", ["미국"], r"americans?"),
    ("British", r"ingiliz(ler\w*)?|british|englishwom[ae]n|englishm[ae]n|brit(isch\w*|in(nen)?|en?)", ["영국"],
     r"british|brits?"),
    ("German", r"alman(lar\w*)?|germans?|deutsch(e|er|en)?", ["독일"], r"germans?"),
    ("French", r"fransız(lar\w*)?|french|französ(isch\w*|in(nen)?|en?)", ["프랑스"], r"french"),
    ("Italian", r"italyan(lar\w*)?|italians?|italien(isch\w*|erin(nen)?|er|ern)", ["이탈리아"], r"italians?"),
    ("Spanish", r"ispanyol(lar\w*)?|spanish|spaniards?|spani(sch\w*|erin(nen)?|er|ern)", ["스페인"], r"spanish|spaniards?"),
    ("Greek", r"yunanl[ıi]\w*|greeks?|griech(isch\w*|in(nen)?|en?)", ["그리스"], r"greeks?"),
    ("Mexican", r"meksikal[ıi]\w*|mexicans?|mexikan\w*", ["멕시코"], r"mexicans?"),
    ("Brazilian", r"brezilyal[ıi]\w*|brazilians?|brasilian\w*", ["브라질"], r"brazilians?"),
    ("Iranian", r"iranl[ıi]\w*|iranians?|persians?|iran(isch\w*|erin(nen)?|er|ern)", ["이란"],
     r"iranians?|persians?"),
    ("Egyptian", r"mısırl[ıi]\w*|egyptians?|ägypt(isch\w*|erin(nen)?|er|ern)", ["이집트"], r"egyptians?"),
    ("Kurdish", r"kürt(ler\w*)?|kurds?|kurdish|kurd(isch\w*|in(nen)?|en?)", ["쿠르드"], r"kurds?|kurdish"),
    ("Scandinavian", r"iskandinav\w*|scandinavians?|nordic|skandinav\w*", ["북유럽"], r"scandinavians?|nordic"),
    ("Thai", r"taylandl[ıi]\w*|thai|thailänd\w*", ["태국"], r"thai"),
    ("Vietnamese", r"vietnaml[ıi]\w*|vietnamese|vietnames\w*", ["베트남"], r"vietnamese"),
    ("Filipino", r"filipinli\w*|filipin[oa]s?|philippin\w*", ["필리핀"], r"filipin[oa]s?"),
    ("Indonesian", r"endonezyal[ıi]\w*|indonesians?|indonesisch\w*", ["인도네시아"], r"indonesians?"),
    ("Caucasian", r"caucasians?", [], r"caucasians?"),
]
_KO_PEOPLE = ("인", " 여", " 남", "계", " 사람", " 소녀", " 소년", " 아이", " 할머니", " 할아버지")
_PERSON_NOUNS = (r"wom[ae]n|m[ae]n|girls?|boys?|lad(?:y|ies)|persons?|people|child(?:ren)?|kids?|guys?|couples?|"
                 r"famil(?:y|ies)|teen(?:ager)?s?|models?|warriors?|soldiers?|students?|"
                 r"grand(?:mother|father|ma|pa)s?|mothers?|fathers?|brides?|grooms?|females?|males?|characters?|"
                 r"faces?|portraits?|businesswom[ae]n|businessm[ae]n|princess(?:es)?|princes?|queens?|kings?|"
                 r"elders?|bab(?:y|ies)|toddlers?|actors?|actress(?:es)?|singers?|dancers?|athletes?|workers?|"
                 r"doctors?|nurses?|chefs?|farmers?|teachers?|monks?|nuns?|samurai|geishas?|ninjas?")
_DEM_ANY = "|".join("(?:%s)" % out for _n, _i, _k, out in _DEMONYMS)
_DEM_OUT = [(name, re.compile(r"\b(?:" + out + r")\b", re.I)) for name, _i, _k, out in _DEMONYMS]
_DEM_IN = [(name, re.compile(r"(?:" + inp + r")", re.I), ko) for name, inp, ko, _o in _DEMONYMS]
_DEM_PEOPLE = re.compile(
    r"\b(?P<d>" + _DEM_ANY + r")"
    r"(?P<mid>(?:[ \-]+(?!(?:and|or|with|in|of|on|at|by|from)\b)(?!(?:" + _DEM_ANY + r")\b)[a-z']+){0,3}?)"
    r"[ \-]+(?P<n>" + _PERSON_NOUNS + r")\b", re.I)
_PERSON_RE = re.compile(r"\b(?:" + _PERSON_NOUNS + r")\b", re.I)


def _tr_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def requested_demonyms(text):
    """Kullanıcının metninde geçen milliyet / köken kelimelerinin İngilizce karşılıkları (metindeki sırayla)."""
    text = text or ""
    low = text.lower()
    words = [(m.start(), m.group(0)) for m in re.finditer(r"[^\W\d_]+", text)]
    found = []
    for name, rx, ko in _DEM_IN:
        pos = [i for i, w in words if rx.fullmatch(w.lower()) or rx.fullmatch(_tr_lower(w))]
        for k in ko:
            for suf in _KO_PEOPLE:
                i = low.find(k + suf)
                if i >= 0:
                    pos.append(i)
        if name == "Middle Eastern":
            m = re.search(r"\bmiddle[ -]east(ern)?\b", low)
            if m:
                pos.append(m.start())
        if pos:
            found.append((min(pos), name))
    return [n for _i, n in sorted(found)]


def _demonym_of(word):
    for name, rx in _DEM_OUT:
        if rx.fullmatch((word or "").strip()):
            return name
    return None


def _needs_an(word):
    w = (word or "").lower()
    if w.startswith(("eu", "uni", "use", "usu", "uto", "one")):
        return False
    if w.startswith(("hour", "honest", "honor", "heir")):
        return True
    return w[:1] in ("a", "e", "i", "o", "u")


def _fix_article(before, next_word):
    """Hemen önce 'a' / 'an' varsa sonraki kelimeye uydurur."""
    m = re.search(r"\b([Aa])n? $", before)
    if not m or not next_word:
        return before
    return before[:m.start()] + m.group(1) + ("n " if _needs_an(next_word) else " ")


def _trim_words(text, limit, tags):
    parts = [p.strip() for p in (text.split(",") if tags else re.split(r"(?<=[.!?])\s+", text)) if p.strip()]
    out, n = [], 0
    for p in parts:
        w = len(p.split())
        if out and n + w > limit:
            break
        out.append(p)
        n += w
    return (", " if tags else " ").join(out)


def check_llm_prompt(model, user_text, extra_text, res):
    """
    LLM'in prompt'unu kullanıcının yazdığına göre denetler:
      - istenen milliyet/köken başka biriyle değiştirilmişse geri koyar ("Turkish woman" → "Asian woman"),
      - kişiler için istenmeyen bir milliyet/köken eklenmişse çıkarır, istenen atlanmışsa ekler,
      - kısa isteğe uzun prompt yazılmışsa kısaltır (ana konu başta olduğu için sondan).
    Döner: (res, notlar); not = (tür, alanlar), tür: swap | removed | added | trimmed
    """
    pos = str(res.get("positive", "")).strip()
    notes = []
    if not pos:
        return res, notes
    tags = model.get("prompt_style") == "tags"
    limit = llm_word_limit(model, user_text)
    n0 = len(pos.split())
    if n0 > int(limit * 1.3):
        short = _trim_words(pos, limit, tags)
        if short and len(short.split()) < n0:
            notes.append(("trimmed", {"n": n0, "m": len(short.split()), "w": len((user_text or "").split())}))
            pos = short
    want = requested_demonyms(user_text)
    allowed = set(want) | set(requested_demonyms(extra_text))
    present = {name for name, rx in _DEM_OUT if rx.search(pos)}
    missing = [w for w in want if w not in present]
    edits = []                                     # (başlangıç, bitiş, yeni metin)
    for m in _DEM_PEOPLE.finditer(pos):
        bad = m.group("d")
        name = _demonym_of(bad)
        if name is None or name in allowed:
            continue
        if missing:
            good = missing.pop(0)
            edits.append((m.start("d"), m.end("d"), good))
            notes.append(("swap", {"bad": bad, "good": good}))
        else:
            end = m.end("d")
            while end < len(pos) and pos[end] in " -":
                end += 1
            edits.append((m.start("d"), end, ""))
            notes.append(("removed", {"bad": bad}))
    for start, end, new in reversed(edits):
        after = pos[end:]
        nxt = new or (after.split()[0] if after.split() else "")
        pos = _fix_article(pos[:start], nxt) + new + after
    for good in missing:                           # istenen ama hiç yazılmamış: ilk kişi kelimesinin önüne
        target = None
        for cand in _PERSON_RE.finditer(pos):
            prev = pos[:cand.start()].rstrip(" -").split()
            if not prev or _demonym_of(prev[-1]) is None:
                target = cand
                break
        if target is None:
            continue
        pos = _fix_article(pos[:target.start()], good) + good + " " + pos[target.start():]
        notes.append(("added", {"good": good}))
    res = dict(res)
    res["positive"] = re.sub(r"[ \t]{2,}", " ", pos).strip()
    return res, notes


# =============================================================================
# Görsel üretim motoru
# =============================================================================
class ImageEngine:
    """
    Tek bir diffusers pipeline'ını yönetir. Aynı model + profil tekrar istenirse yeniden yüklemez.
    'full' profilde LLM çalışmadan önce pipeline geçici olarak RAM'e taşınır (VRAM çakışması olmasın).
    İkinci geçiş (hires fix) ve yüz/el düzeltici, AYNI bileşenleri paylaşan bir img2img pipeline'ı kullanır:
    ek VRAM/RAM gerekmez, model ikinci kez yüklenmez.
    """

    def __init__(self):
        self.pipe = None
        self.key = None
        self.profile = None
        self.ram_tight = False
        self.device = "cpu"
        self.parked = False
        self.sampler = "default"
        self._orig_sched = None
        self._variant = None
        self._loras = {}            # adapter adı -> dosya yolu
        self.lock = threading.RLock()

    def unload(self):
        with self.lock:
            self.pipe = None
            self._variant = None
            self._orig_sched = None
            self._loras = {}
            self.sampler = "default"
            self.key = None
            self.profile = None
            self.parked = False
            free_gpu_memory()

    def park(self):
        """VRAM'i boşalt (sadece tam GPU profilinde gerekir)."""
        with self.lock:
            if self.pipe is not None and self.profile == "full" and not self.parked:
                self.pipe.to("cpu")
                self.parked = True
                free_gpu_memory()

    def unpark(self):
        with self.lock:
            if self.pipe is not None and self.parked:
                self.pipe.to(self.device)
                self.parked = False

    def load(self, model, path, profile, hw):
        with self.lock:
            if self.pipe is not None and self.key == model["key"] and self.profile == profile:
                self.unpark()
                return False
            self.unload()

            disk_gb = folder_size_bytes(path) / GB
            need = model_ram_need_gb(model, profile, disk_gb)
            ok, usable, tight = ram_can_load(need, hw.ram_total_gb)
            if not ok:
                raise MemoryGuardError((need, usable))
            self.ram_tight = tight

            torch = import_torch()
            try:
                import diffusers
                from diffusers import DiffusionPipeline
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa
                raise EngineBrokenError(f"diffusers: {type(e).__name__}: {str(e)[:200]}") from e

            if not model.get("custom"):
                with open(os.path.join(path, "model_index.json"), encoding="utf-8") as f:
                    cls_name = json.load(f).get("_class_name", "")
                if cls_name and not hasattr(diffusers, cls_name):
                    raise EngineBrokenError(f"diffusers {getattr(diffusers, '__version__', '?')} lacks {cls_name}")

            on_gpu = profile != "cpu" and hw.cuda_usable
            if model["dtype"] == "fp16":
                dtype = torch.float16 if on_gpu else torch.float32
            else:
                dtype = torch.bfloat16
            kwargs = dict(torch_dtype=dtype)
            kwargs.update(model.get("load_kwargs", {}))
            variant = model.get("variant")
            if model.get("custom"):
                pipe = load_custom_pipeline(model, kwargs)
            else:
                try:
                    pipe = DiffusionPipeline.from_pretrained(path, variant=variant, **kwargs) if variant \
                        else DiffusionPipeline.from_pretrained(path, **kwargs)
                except (ValueError, OSError) as e:
                    if variant and "variant" in str(e).lower():
                        pipe = DiffusionPipeline.from_pretrained(path, **kwargs)
                    else:
                        raise

            dev = hw.device if on_gpu else "cpu"
            if profile == "full" and on_gpu:
                pipe.to(dev)
            elif profile in ("normal", "balanced") and on_gpu:
                pipe.enable_model_cpu_offload(device=dev)
            elif profile == "low" and on_gpu:
                pipe.enable_sequential_cpu_offload(device=dev)
            else:
                pipe.to("cpu")

            vae = getattr(pipe, "vae", None)
            if vae is not None:
                if hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                if profile in ("balanced", "low", "cpu") and hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
            if profile in ("balanced", "low") and hasattr(pipe, "unet"):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

            self.pipe, self.key, self.profile, self.device, self.parked = pipe, model["key"], profile, dev, False
            self._orig_sched = getattr(pipe, "scheduler", None)
            self.sampler = "default"
            self._variant = None
            self._loras = {}
            return True

    # ------------------------------------------------------------------ LoRA
    def _present_adapters(self):
        """Pipeline'da gerçekten yüklü olan LoRA adaptörlerinin adları."""
        try:
            got = self.pipe.get_list_adapters() or {}
            return {n for names in got.values() for n in (names or [])}
        except Exception:
            return set(self._loras)

    def _unload_all_loras(self):
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
        self._loras = {}

    def set_loras(self, specs):
        """
        specs: [(kimlik, dosya yolu, güç)] — yalnızca bu modelle uyumlu olanlar.
        Değişmeyen LoRA'lar yeniden yüklenmez; yalnızca güçleri güncellenir.
        Dönüş: (uygulananlar [(kimlik, güç)], hatalar [(kimlik, mesaj)]).
        """
        import hashlib
        with self.lock:
            if self.pipe is None:
                raise RuntimeError("pipeline not loaded")
            want = {}
            for key, path, weight in specs:
                name = "l" + hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
                want[name] = (path, float(weight), key)
            if not want:
                if self._loras:
                    self.pipe.unload_lora_weights()
                    self._loras = {}
                return [], []
            present = self._present_adapters()
            for n in [n for n in self._loras if n not in present]:
                self._loras.pop(n, None)          # kayıtta var ama modelde yok: izi temizle
            stale = [n for n in self._loras if n not in want]
            if stale:
                try:
                    self.pipe.delete_adapters(stale)
                    for n in stale:
                        self._loras.pop(n, None)
                except Exception:
                    self._unload_all_loras()
            errors = []
            for n, (path, _w, key) in want.items():
                if n in self._loras:
                    continue
                try:
                    self.pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path), adapter_name=n)
                except Exception as e:
                    errors.append((key, f"{type(e).__name__}: {str(e)[:180]}"))
                    continue
                # diffusers, modelde karşılığı olmayan bir LoRA'yı hata vermeden "yükler" ama adaptör oluşturmaz
                # (başka bir modele ait LoRA, desteklenmeyen biçim…). Bu durumda LoRA atlanır, iş çökmez.
                if n in self._present_adapters():
                    self._loras[n] = path
                else:
                    errors.append((key, "no matching layers in this model (LoRA made for another model or an "
                                        "unsupported LoRA type)"))
            names = [n for n in want if n in self._loras]
            if names:
                try:
                    self.pipe.set_adapters(names, adapter_weights=[want[n][1] for n in names])
                except Exception as e:
                    self._unload_all_loras()
                    errors += [(want[n][2], f"{type(e).__name__}: {str(e)[:180]}") for n in names]
                    names = []
            elif self._loras:
                self._unload_all_loras()
            return [(want[n][2], want[n][1]) for n in names], errors

    # ------------------------------------------------------------------ örnekleyici (sampler)
    def apply_sampler(self, model, name):
        """SD 1.5 / SDXL için örnekleyiciyi ayarlar. Dönüş: gerçekten kullanılan örnekleyici adı."""
        with self.lock:
            if self.pipe is None or not model.get("samplers") or self._orig_sched is None:
                return "default"
            if name not in SAMPLERS or name == "auto":
                name = "dpmpp_2m_karras"
            if name == self.sampler:
                return name
            import diffusers
            cfg = self._orig_sched.config
            try:
                if name == "default":
                    sch = self._orig_sched
                elif name == "dpmpp_2m_karras":
                    sch = diffusers.DPMSolverMultistepScheduler.from_config(
                        cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++", solver_order=2,
                        euler_at_final=(model_family(model) == "sdxl"))
                elif name == "euler_a":
                    sch = diffusers.EulerAncestralDiscreteScheduler.from_config(cfg)
                elif name == "euler":
                    sch = diffusers.EulerDiscreteScheduler.from_config(cfg)
                else:
                    sch = diffusers.UniPCMultistepScheduler.from_config(cfg)
            except Exception:
                sch, name = self._orig_sched, "default"
            self.pipe.scheduler = sch
            if self._variant is not None:
                self._variant.scheduler = sch
            self.sampler = name
            return name

    # ------------------------------------------------------------------ üretim
    def generate(self, model, prompt, negative, width, height, steps, cfg, seed, step_cb, abort_event,
                 sampler="auto"):
        torch = import_torch()
        with self.lock:
            pipe = self.pipe
            if pipe is None:
                raise RuntimeError("pipeline not loaded")
            self.apply_sampler(model, sampler)
            self._tiling_for(width, height)
            kwargs = dict(prompt=prompt, width=width, height=height, num_inference_steps=steps,
                          guidance_scale=cfg, generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                          num_images_per_prompt=1, callback_on_step_end=self._make_cb(steps, step_cb, abort_event))
            if model["neg"] and negative:
                kwargs["negative_prompt"] = negative
            with torch.inference_mode():
                out = pipe(**kwargs)
            return out.images[0]

    def _tiling_for(self, width, height):
        if self.profile in ("balanced", "low", "cpu") or width * height > 1024 * 1024 * 1.3:
            vae = getattr(self.pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

    @staticmethod
    def _make_cb(steps, step_cb, abort_event):
        def _cb(p, step, timestep, kw):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            if step_cb:
                # Bazı zamanlayıcılar (ör. SD 1.5'teki PNDM) ve img2img istenenden farklı sayıda adım atar.
                total = getattr(p, "_num_timesteps", None) or steps
                step_cb(min(step + 1, total), total)
            return kw
        return _cb

    # ------------------------------------------------------------------ img2img (hires fix / düzeltici)
    def supports_refine(self, model):
        try:
            import diffusers
        except BaseException:  # noqa
            return False
        return bool(model.get("img2img")) and hasattr(diffusers, model["img2img"])

    def _img2img(self, model):
        """Yüklü pipeline'ın bileşenlerini paylaşan img2img pipeline'ı (önbellekli). from_pipe KULLANILMAZ:
        o, bileşenleri float32'ye çevirip belleği ikiye katlayabilir."""
        if self._variant is not None:
            return self._variant
        import inspect
        import diffusers
        cls = getattr(diffusers, model.get("img2img") or "", None)
        if cls is None:
            raise RuntimeError("img2img pipeline not available in this diffusers version")
        params = inspect.signature(cls.__init__).parameters
        kw = {k: v for k, v in self.pipe.components.items() if k in params}
        cfg = dict(getattr(self.pipe, "config", {}) or {})
        for k in params:
            if k != "self" and k not in kw and k in cfg and not isinstance(cfg[k], (list, tuple, dict)):
                kw[k] = cfg[k]
        if "requires_safety_checker" in params:
            kw["requires_safety_checker"] = False
        v = cls(**kw)
        try:
            v.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self._variant = v
        return v

    def refine(self, model, image, prompt, negative, strength, steps, cfg, seed, step_cb, abort_event):
        """
        Görseli aynı modelle düşük güçte yeniden çizer (img2img). Görselin boyutu korunur
        (boyut 16'nın katı olmalı). Model CPU-offload profillerinde iş bitince bileşenler RAM'e geri alınır.
        """
        import inspect
        torch = import_torch()
        from PIL import Image
        with self.lock:
            if self.pipe is None:
                raise RuntimeError("pipeline not loaded")
            pipe = self._img2img(model)
            w, h = image.size
            self._tiling_for(w, h)
            params = inspect.signature(pipe.__call__).parameters
            kwargs = dict(prompt=prompt, image=image.convert("RGB"), strength=float(strength),
                          num_inference_steps=int(steps), guidance_scale=cfg,
                          generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                          num_images_per_prompt=1, callback_on_step_end=self._make_cb(steps, step_cb, abort_event))
            if "width" in params and "height" in params:
                kwargs.update(width=w, height=h)
            if "mask_image" in params:          # inpaint pipeline'ı: beyaz maske = tüm kırpıntıyı yeniden çiz
                kwargs["mask_image"] = Image.new("L", (w, h), 255)
            if model["neg"] and negative and "negative_prompt" in params:
                kwargs["negative_prompt"] = negative
            try:
                with torch.inference_mode():
                    out = pipe(**kwargs)
            finally:
                # Ortak bileşenlerdeki offload kancalarını sıfırla: hepsi RAM'e döner, VRAM boşalır.
                if self.profile in ("normal", "balanced") and hasattr(self.pipe, "maybe_free_model_hooks"):
                    try:
                        self.pipe.maybe_free_model_hooks()
                    except Exception:
                        pass
            res = out.images[0]
            if res.size != (w, h):
                res = res.resize((w, h), Image.LANCZOS)
            return res


# =============================================================================
# Real-ESRGAN x2 (RRDBNet) — basicsr'a bağımlı olmadan, karo karo (düşük VRAM dostu)
# =============================================================================
_ESRGAN_NET = {"net": None, "device": None}


def _build_rrdbnet(scale=2, num_feat=64, num_block=23, num_grow_ch=32):
    torch = import_torch()
    from torch import nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf, gc):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, gc):
            super().__init__()
            self.rdb1 = RDB(nf, gc)
            self.rdb2 = RDB(nf, gc)
            self.rdb3 = RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            in_ch = 3 * (4 if scale == 2 else (16 if scale == 1 else 1))
            self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, 2)
            elif self.scale == 1:
                feat = F.pixel_unshuffle(x, 4)
            else:
                feat = x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def load_upscaler(weights_path, device):
    torch = import_torch()
    if _ESRGAN_NET["net"] is not None and _ESRGAN_NET["device"] == device:
        return _ESRGAN_NET["net"]
    net = _build_rrdbnet(scale=2)
    # weights_only=True: dosyanın içinde kod çalıştırılamaz (güvenli yükleme).
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict):
        sd = sd.get("params_ema") or sd.get("params") or sd
    net.load_state_dict(sd, strict=True)
    net.eval()
    use_half = device.startswith("cuda")
    net = net.to(device, dtype=torch.float16 if use_half else torch.float32)
    _ESRGAN_NET.update(net=net, device=device)
    return net


def unload_upscaler():
    _ESRGAN_NET.update(net=None, device=None)
    free_gpu_memory()


def upscale_2x(img, net, device, tile=256, pad=16, progress_cb=None, abort_event=None):
    import numpy as np
    torch = import_torch()
    import torch.nn.functional as F
    from PIL import Image

    img = img.convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    h0, w0 = x.shape[2], x.shape[3]
    # pixel_unshuffle çift boyut ister
    ph, pw = h0 % 2, w0 % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    h, w = x.shape[2], x.shape[3]
    out = torch.zeros((1, 3, h * 2, w * 2), dtype=torch.float32)
    dtype = next(net.parameters()).dtype
    tiles = [(ty, tx) for ty in range(0, h, tile) for tx in range(0, w, tile)]
    with torch.inference_mode():
        for n, (y0, x0) in enumerate(tiles):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            # tile ve pad çift, görüntü de çifte tamamlandı → karo kenarları her zaman çift olur
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            t = x[:, :, py0:py1, px0:px1].to(device, dtype=dtype)
            o = net(t).float().cpu()
            oy, ox = (y0 - py0) * 2, (x0 - px0) * 2
            out[:, :, y0 * 2:y1 * 2, x0 * 2:x1 * 2] = o[:, :, oy:oy + (y1 - y0) * 2, ox:ox + (x1 - x0) * 2]
            if progress_cb:
                progress_cb(n + 1, len(tiles))
    out = out[:, :, :h0 * 2, :w0 * 2].clamp_(0, 1)
    res = (out[0].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(res)


# =============================================================================
# Yüz / el düzeltici (ADetailer benzeri) — İSTEĞE BAĞLI bileşen, yalnızca kullanıcı onayıyla kurulur
#   Algılama: OpenCV (Apache-2.0) + YuNet yüz modeli (MIT) + MediaPipe avuç içi modeli (Apache-2.0)
#             + anime/manga yüz kaskadı lbpcascade_animeface (MIT). Hepsi ticari kullanıma uygundur.
#   Düzeltme: bulunan her yüz/el kırpılır, modelin doğal çözünürlüğüne büyütülür, aynı görsel modeliyle
#             düşük güçte yeniden çizilir (img2img) ve yumuşak kenarlı maskeyle yerine yapıştırılır.
#   Dosyalar commit'e sabitlenmiş adreslerden indirilir ve SHA-256 ile doğrulanır.
# =============================================================================
DETECTOR_PIP = ["opencv-python-headless==4.14.0.94"]   # 5.x serisinde CascadeClassifier yok
DETECTOR_PIP_FALLBACK = ["opencv-python-headless==4.10.0.84"]   # numpy 1.x olan ortamlar için
_ZOO = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/")
DETECTOR_FILES = [
    {"key": "face", "file": "face_detection_yunet_2023mar.onnx", "license": "MIT", "size": 232589,
     "url": _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"},
    {"key": "hand", "file": "palm_detection_mediapipe_2023feb.onnx", "license": "Apache-2.0", "size": 3905734,
     "url": _ZOO + "palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
     "sha256": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"},
    {"key": "anime", "file": "lbpcascade_animeface.xml", "license": "MIT", "size": 246945,
     "url": ("https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/"
             "4433ab1ae1166ea75acfe99eb0f18709dac329a0/lbpcascade_animeface.xml"),
     "sha256": "9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30"},
]
DETAILER = {"key": "detailer", "name": "Face & Hand Fixer (OpenCV + YuNet + MediaPipe Palm)",
            "short": "Face & Hand Fixer", "size": "~50 MB", "size_gb": 0.2,
            "license": "Apache-2.0 / MIT", "desc": "desc_detailer"}

# Ortak "kötü anatomi" negatifleri (negatif prompt destekleyen SD 1.5 / SDXL için).
ANATOMY_NEGATIVE = ("worst quality, low quality, lowres, blurry, jpeg artifacts, bad anatomy, bad proportions, "
                    "deformed, disfigured, poorly drawn face, ugly face, asymmetric eyes, cross-eyed, "
                    "bad hands, poorly drawn hands, mutated hands, extra fingers, missing fingers, fused fingers, "
                    "too many fingers, extra limbs, extra arms, missing arms, long neck, watermark, text, signature")
FACE_FIX_POS = "detailed face, beautiful detailed eyes, sharp focus"
HAND_FIX_POS = "detailed hand, five fingers, correct hand anatomy, sharp focus"
FACE_FIX_NEG = "deformed face, ugly face, asymmetric eyes, blurry face"
HAND_FIX_NEG = "bad hands, extra fingers, missing fingers, fused fingers, mutated hands, deformed hand"


def merge_prompt_lists(first, second):
    """Virgülle ayrılmış iki listeyi birleştirir; tekrar edenleri atar, ilk listenin sırası korunur."""
    items, seen = [], set()
    for part in ((first or "") + "," + (second or "")).split(","):
        s = part.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            items.append(s)
    return ", ".join(items)


def detector_dir(models_root):
    return os.path.join(models_root, "detect")


def detector_files_ok(models_root):
    d = detector_dir(models_root)
    for f in DETECTOR_FILES:
        p = os.path.join(d, f["file"])
        if not os.path.exists(p) or os.path.getsize(p) != f["size"]:
            return False
    return True


def cv2_installed(paths):
    return os.path.exists(os.path.join(paths.site_packages(), "cv2", "__init__.py"))


def is_detailer_installed(paths):
    return cv2_installed(paths) and detector_files_ok(paths.models)


_CV2_VERIFY = ("import json\ntry:\n    import cv2\n    print('AILS_CV2 ' + json.dumps({'ok': hasattr(cv2, 'FaceDetectorYN') "
               "and hasattr(cv2, 'dnn') and hasattr(cv2, 'CascadeClassifier'), 'ver': cv2.__version__}))\n"
               "except BaseException as e:\n    print('AILS_CV2 ' + json.dumps({'ok': False, 'err': (type(e).__name__ "
               "+ ': ' + str(e))[:300]}))\n")


def verify_cv2(paths, timeout=300):
    """OpenCV'yi ayrı süreçte sınar (ana süreç bozuk bir kurulumla kirlenmesin)."""
    try:
        r = subprocess.run([paths.venv_python, "-c", _CV2_VERIFY], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=sub_env(paths), **no_window_kwargs())
    except Exception as e:
        return {"ok": False, "err": str(e)}
    for line in (r.stdout or "").splitlines()[::-1]:
        if line.startswith("AILS_CV2 "):
            try:
                return json.loads(line[9:])
            except Exception:
                break
    return {"ok": False, "err": ((r.stderr or "") + (r.stdout or "")).strip()[-300:]}


CV2_SIZE = "~40 MB"


def install_cv2(paths, log_cb, msg, progress_cb=None, abort_event=None):
    """OpenCV'yi (yalnızca bu paket — '--no-deps' ile, motorun numpy/torch'una dokunmaz) programın ortamına kurar.
    Yüz/el düzeltici ve videodan kare çıkarma bunu kullanır."""
    if cv2_installed(paths) and verify_cv2(paths).get("ok"):
        return True
    check_space(paths.root, 0.5)
    log_cb(msg("log_det_pip"))
    args = ["--no-deps", "--only-binary=:all:"]
    status, out = run_pip(paths, args + DETECTOR_PIP, log_cb, abort_event, progress_cb)
    raise_for_pip(status, paths, 0.5, out)
    v = verify_cv2(paths)
    if not v.get("ok"):
        log_cb("   OpenCV: " + str(v.get("err", ""))[:200])
        status, out = run_pip(paths, args + ["--force-reinstall"] + DETECTOR_PIP_FALLBACK, log_cb, abort_event,
                              progress_cb)
        raise_for_pip(status, paths, 0.5, out)
        v = verify_cv2(paths)
        if not v.get("ok"):
            raise RuntimeError("OpenCV: " + str(v.get("err", "import failed")))
    log_cb(msg("log_det_cv2", ver=v.get("ver", "?")))
    importlib.invalidate_caches()
    return True


def install_detector(paths, log_cb, msg, progress_cb=None, abort_event=None):
    """
    Yüz/el düzelticiyi kurar: OpenCV + üç küçük algılama modeli. Hepsi veri klasörüne gider.
    """
    check_space(paths.root, 0.5)
    try:
        install_cv2(paths, log_cb, msg, progress_cb, abort_event)
        d = detector_dir(paths.models)
        for f in DETECTOR_FILES:
            dest = os.path.join(d, f["file"])
            if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
                continue
            log_cb(msg("log_det_file", name=f["file"], lic=f["license"]))
            download_file(f["url"], dest, progress_cb, abort_event, need_gb=0.05, sha256=f["sha256"])
    finally:
        clean_dir(paths.tmp)
    importlib.invalidate_caches()
    _DETECTOR.update(obj=None, dir=None)
    return True


def remove_detector(paths):
    """Algılama modellerini siler (OpenCV paketi küçük olduğundan ortamda bırakılır)."""
    _DETECTOR.update(obj=None, dir=None)
    delete_path(detector_dir(paths.models))


def _palm_anchors():
    """MediaPipe avuç içi algılayıcısının 2016 SSD çapası (192×192 giriş; 8 ve 16 piksel adım)."""
    import numpy as np
    pts = []
    for stride, per_cell in ((8, 2), (16, 6)):
        n = 192 // stride
        for y in range(n):
            for x in range(n):
                pts.extend([((x + 0.5) / n, (y + 0.5) / n)] * per_cell)
    return np.array(pts, dtype=np.float32)


def _overlap(a, b):
    """İki kutunun kesişimi / küçük kutunun alanı (iç içe kutuları da yakalar)."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    small = min(a[2] * a[3], b[2] * b[3])
    return (ix * iy) / small if small > 0 else 0.0


def _suppress(dets, thr):
    """Skoru yüksek olan kalır; onunla çok örtüşenler atılır. dets: [(x, y, w, h, skor, ...)]"""
    out = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(_overlap(d, o) < thr for o in out):
            out.append(d)
    return out


class FaceHandDetector:
    """CPU'da çalışır (GPU belleği kullanmaz); 1024 px bir görselde toplam ~0.1–0.5 sn."""

    def __init__(self, det_dir):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        f = {x["key"]: os.path.join(det_dir, x["file"]) for x in DETECTOR_FILES}
        self.yunet = cv2.FaceDetectorYN.create(f["face"], "", (320, 320), 0.5, 0.3, 200)
        self.palm = cv2.dnn.readNet(f["hand"])
        cas = cv2.CascadeClassifier(f["anime"])
        self.anime = None if cas.empty() else cas
        self.anchors = _palm_anchors()

    # ---- yüzler (gerçekçi + anime) ----
    def faces(self, rgb, score=0.5):
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        s = min(1.0, 1600.0 / max(h, w))
        img = bgr if s >= 1.0 else cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                                               interpolation=cv2.INTER_AREA)
        self.yunet.setScoreThreshold(float(score))
        self.yunet.setInputSize((img.shape[1], img.shape[0]))
        _ok, res = self.yunet.detect(img)
        dets = []
        if res is not None:
            for r in res:
                x, y, bw, bh = [float(v) / s for v in r[:4]]
                dets.append((x, y, bw, bh, float(r[-1]), "face"))
        if self.anime is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            mn = max(24, int(min(h, w) * 0.05))
            found = self.anime.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(mn, mn))
            for (x, y, bw, bh) in (found if len(found) else []):
                dets.append((float(x), float(y), float(bw), float(bh), 0.45, "face"))
        return _suppress(dets, 0.4)

    # ---- eller (avuç içi algılama → tüm eli kapsayan kare bölge) ----
    def _palm_pass(self, rgb, score):
        cv2, np = self.cv2, self.np
        th, tw = rgb.shape[:2]
        ratio = 192.0 / max(th, tw)
        nw, nh = max(1, int(tw * ratio)), max(1, int(th * ratio))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        left, top = (192 - nw) // 2, (192 - nh) // 2
        img = cv2.copyMakeBorder(img, top, 192 - nh - top, left, 192 - nw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.palm.setInput((img.astype(np.float32) / 255.0)[np.newaxis])
        outs = self.palm.forward(self.palm.getUnconnectedOutLayersNames())
        reg = next(o for o in outs if o.shape[-1] == 18)[0]
        raw = next(o for o in outs if o.shape[-1] == 1)[0, :, 0].astype(np.float64)
        sc = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        scale = max(th, tw)
        bias = np.array([left / ratio, top / ratio], dtype=np.float32)
        dets = []
        for i in np.where(sc >= score)[0]:
            d = reg[i]
            c = (d[0:2] / 192.0 + self.anchors[i]) * scale - bias
            wh = d[2:4] / 192.0 * scale
            kps = (d[4:18].reshape(7, 2) / 192.0 + self.anchors[i]) * scale - bias
            dets.append((float(c[0] - wh[0] / 2), float(c[1] - wh[1] / 2), float(wh[0]), float(wh[1]),
                         float(sc[i]), kps))
        return _suppress(dets, 0.3)

    def hands(self, rgb, score=0.5):
        np = self.np
        h, w = rgb.shape[:2]
        tiles = [(0, 0, w, h)]
        if min(h, w) >= 320:            # küçük eller için ayrıca örtüşen karolarda ara
            tw, th = int(w * 0.6), int(h * 0.6)
            for x0, y0 in ((0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th), ((w - tw) // 2, (h - th) // 2)):
                tiles.append((x0, y0, tw, th))
        palms = []
        for x0, y0, tw, th in tiles:
            for d in self._palm_pass(rgb[y0:y0 + th, x0:x0 + tw], score):
                palms.append((d[0] + x0, d[1] + y0, d[2], d[3], d[4], d[5] + np.array([x0, y0], dtype=np.float32)))
        out = []
        for x, y, bw, bh, s, kps in _suppress(palms, 0.4):
            size = max(bw, bh)
            v = kps[2] - kps[0]                     # bilek → orta parmak kökü (parmakların yönü)
            n = float(np.hypot(v[0], v[1])) or 1.0
            cx = x + bw / 2 + v[0] / n * size * 0.5
            cy = y + bh / 2 + v[1] / n * size * 0.5
            side = size * 2.3
            out.append((cx - side / 2, cy - side / 2, side, side, s, "hand"))
        return _suppress(out, 0.5)


_DETECTOR = {"obj": None, "dir": None}


def get_detector(models_root):
    d = detector_dir(models_root)
    if _DETECTOR["obj"] is None or _DETECTOR["dir"] != d:
        _DETECTOR.update(obj=FaceHandDetector(d), dir=d)
    return _DETECTOR["obj"]


def _clip_box(x, y, w, h, W, H):
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def plan_detail_regions(dets, img_w, img_h, base_px, max_each=6):
    """
    Algılanan kutulardan düzeltme bölgeleri üretir: [{"kind", "crop": (x,y,w,h), "mask": (x,y,w,h), "score"}]
    Çok küçük (<12 px) ya da modelin doğal çözünürlüğünün iki katından büyük bölgeler atlanır.
    """
    regions = []
    for kind, ctx, msk in (("face", 1.9, 1.25), ("hand", 1.5, 1.08)):
        ds = sorted((d for d in dets if d[5] == kind), key=lambda d: -(d[2] * d[3]))[:max_each]
        for x, y, w, h, s, _k in ds:
            if min(w, h) < 12:
                continue
            cx, cy = x + w / 2, y + h / 2
            side = max(w, h) * ctx
            crop = _clip_box(cx - side / 2, cy - side / 2, side, side, img_w, img_h)
            mask = _clip_box(cx - w * msk / 2, cy - h * msk / 2, w * msk, h * msk, img_w, img_h)
            if crop[2] < 16 or crop[3] < 16 or max(crop[2], crop[3]) > base_px * 2:
                continue
            regions.append({"kind": kind, "crop": crop, "mask": mask, "score": s})
    return regions


def detail_work_size(cw, ch, base_px):
    """Kırpıntının işleneceği boyut: uzun kenar en az modelin doğal çözünürlüğü, 16'nın katı."""
    long_side = max(cw, ch)
    target = max(base_px, min(long_side, int(base_px * 1.5)))
    s = target / float(long_side)
    pw = max(64, int(round(cw * s / 16.0)) * 16)
    ph = max(64, int(round(ch * s / 16.0)) * 16)
    return pw, ph


def paste_region(dst, patch, region):
    """İşlenmiş kırpıntıyı yumuşak kenarlı maskeyle görsele geri yapıştırır (yerinde değiştirir)."""
    from PIL import Image, ImageDraw, ImageFilter
    cx, cy, cw, ch = region["crop"]
    mx, my, mw, mh = region["mask"]
    patch = patch.convert("RGB").resize((cw, ch), Image.LANCZOS)
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    feather = max(2, int(0.08 * max(mw, mh)))
    box = (mx - cx + feather // 2, my - cy + feather // 2, mx - cx + mw - feather // 2, my - cy + mh - feather // 2)
    if box[2] <= box[0] or box[3] <= box[1]:
        return dst
    if region["kind"] == "face":
        draw.ellipse(box, fill=255)
    else:
        draw.rounded_rectangle(box, radius=max(2, int(min(mw, mh) * 0.2)), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather / 2.0))
    base = dst.crop((cx, cy, cx + cw, cy + ch)).convert("RGB")
    dst.paste(Image.composite(patch, base, mask), (cx, cy))
    return dst


FACE_SCORE = 0.5      # YuNet çok isabetli; bozuk yüzleri de yakalamak için eşik düşük tutulur
HAND_SCORE = 0.6      # avuç içi algılayıcısı 0.6 altında ayakkabı/doku gibi şeyleri de el sanabiliyor


def detect_regions(models_root, img, base_px, faces=True, hands=True, max_each=6):
    """Görseldeki yüz ve elleri bulur; düzeltme bölgelerini döndürür."""
    import numpy as np
    det = get_detector(models_root)
    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    dets = []
    if faces:
        dets += det.faces(rgb, FACE_SCORE)
    if hands:
        dets += det.hands(rgb, HAND_SCORE)
    return plan_detail_regions(dets, img.width, img.height, base_px, max_each=max_each)


# =============================================================================
# İkinci geçiş (hires fix) ve yüz/el düzeltme — ImageEngine.refine üzerinden
# =============================================================================
HAND_FIX_GENERIC = "detailed, sharp focus, correct anatomy"


def run_hires(engine, model, img, prompt, negative, steps, cfg, seed, step_cb, abort_event, scale=1.5):
    """Görseli ~1.5 kat büyütüp aynı modelle düşük güçte yeniden çizer. Dönüş: (görsel, yeni_boyut | None)."""
    from PIL import Image
    size = hires_size(img.width, img.height, model, scale)
    if size is None:
        return img, None
    up = img.convert("RGB").resize(size, Image.LANCZOS)
    st = float(model.get("hires_strength", 0.45))
    out = engine.refine(model, up, prompt, negative, st, refine_steps(model, steps, st), cfg, seed, step_cb, abort_event)
    return out, size


def run_detailer(engine, model, img, models_root, prompt, negative, steps, cfg, seed, faces=True, hands=True,
                 region_cb=None, step_cb=None, abort_event=None, regions=None):
    """
    Yüz/el düzeltici. Dönüş: (görsel, {"face": n, "hand": m}).
    Algılayıcı yanlışlıkla bir eşyayı el sanarsa ona el çizilmesin diye el bölgelerinde prompt'a
    "el" kelimesi eklenmez; düzeltmeyi yüksek çözünürlükte yeniden çizim ve negatifler sağlar.
    """
    from PIL import Image
    if regions is None:
        regions = detect_regions(models_root, img, model["base_px"], faces=faces, hands=hands)
    counts = {"face": 0, "hand": 0}
    out = img.convert("RGB").copy()
    for k, r in enumerate(regions):
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
        if region_cb:
            region_cb(k + 1, len(regions), r["kind"])
        cx, cy, cw, ch = r["crop"]
        work = out.crop((cx, cy, cx + cw, cy + ch)).resize(detail_work_size(cw, ch, model["base_px"]), Image.LANCZOS)
        if r["kind"] == "face":
            pos, neg = FACE_FIX_POS + ", " + prompt, merge_prompt_lists(FACE_FIX_NEG, negative)
        else:
            pos, neg = HAND_FIX_GENERIC + ", " + prompt, merge_prompt_lists(HAND_FIX_NEG, negative)
        st = float(model.get("detail_strength", {}).get(r["kind"], 0.4))
        res = engine.refine(model, work, pos, neg, st, refine_steps(model, steps, st), cfg,
                            (int(seed) + 101 + k) % (2 ** 31 - 1), step_cb, abort_event)
        paste_region(out, res, r)
        counts[r["kind"]] += 1
    return out, counts


# =============================================================================
# v2 hazırlığı — Stil Öğrenme / Görsel Analiz veri setleri
#   Görseller kullanıcının seçtiği klasör/dosyalardan ya da (kullanıcı açıkça izin verirse) doğrudan görsel
#   adreslerinden alınır; veri klasöründeki datasets\<ad>\images içine kopyalanır. Analiz motoru v2'de gelir.
#   İnternet kuralları: tek bağlantı, sırayla, istekler arası en az 1 sn, en fazla 500 adres ve 25 MB/görsel.
# =============================================================================
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DATASET_FILE = "dataset.json"
WEB_MAX_PER_BATCH = 500
WEB_MAX_MB = 25
WEB_MIN_DELAY = 1.0
# Wikimedia vb. sunucular, kim olduğunu ve iletişim adresini söyleyen bir User-Agent ister.
APP_UA = "AILocalImageStudio/1.4.3 (+https://github.com/etemcankvrc/AI-Local-Image-Studio; one request per second)"

# Openverse (WordPress): Wikimedia, Flickr, müzeler vb. kaynaklardaki AÇIK LİSANSLI görsellerde arama.
# Anahtarsız kullanım sınırları (Openverse kaynak kodundaki varsayılanlar): sayfa başına en fazla 20 sonuç,
# bir aramada en fazla 240 sonuç, saatte 5 / günde 100 arama isteği. Sınırları aşmaya çalışılmaz
# (ör. 'openverse.org' gibi görünmek yasak); 429 gelince arama durur.
OPENVERSE_API = "https://api.openverse.org/v1/images/"
OPENVERSE_PAGE = 20
OPENVERSE_MAX_PAGES = 12
OPENVERSE_LICENSES = {"commercial": "cc0,pdm,by", "commercial_sa": "cc0,pdm,by,by-sa"}
OPENVERSE_CATEGORIES = ("all", "illustration", "digitized_artwork", "photograph")
SEARCH_COUNTS = (20, 40, 60, 100)
_WIN_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


class OfflineError(Exception):
    """Çevrimdışı mod açıkken internete çıkılmak istendi."""


class RateLimitedError(Exception):
    """Arama servisi kullanım sınırına ulaşıldı (HTTP 429)."""


def datasets_root(paths):
    return os.path.join(paths.root, "datasets")


def addons_dir(paths):
    return os.path.join(paths.root, "addons")


def list_addons(paths):
    d = addons_dir(paths)
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(".safetensors"))
    except OSError:
        return []


def safe_folder_name(name, fallback="dataset"):
    """Kullanıcının verdiği adı (Türkçe/Korece harfler korunarak) geçerli bir klasör adına çevirir."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    s = re.sub(r"\s+", "_", s).strip(" ._")[:60]
    if not s or s.lower() in _WIN_RESERVED:
        s = fallback
    return s


def _ds_dir(paths, slug):
    return os.path.join(datasets_root(paths), slug)


def load_dataset(paths, slug):
    try:
        with open(os.path.join(_ds_dir(paths, slug), DATASET_FILE), encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            d.setdefault("items", [])
            d.setdefault("name", slug)
            d.setdefault("tag", "")
            d["learn"] = norm_learn(d.get("learn"))
            return d
    except Exception:
        pass
    return None


def save_dataset(paths, slug, data):
    d = _ds_dir(paths, slug)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, DATASET_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(d, DATASET_FILE))


def create_dataset(paths, name, tag="", learn="style"):
    base = safe_folder_name(name)
    slug, n = base, 2
    while os.path.exists(_ds_dir(paths, slug)):
        slug, n = f"{base}_{n}", n + 1
    os.makedirs(os.path.join(_ds_dir(paths, slug), "images"), exist_ok=True)
    save_dataset(paths, slug, {"name": (name or slug).strip() or slug, "tag": (tag or "").strip(), "items": [],
                               "learn": norm_learn(learn), "created": time.strftime("%Y-%m-%d %H:%M"), "version": 1})
    return slug


def delete_dataset(paths, slug):
    delete_path(_ds_dir(paths, slug))


def dataset_stats(paths, slug, data=None):
    data = data or load_dataset(paths, slug) or {"items": []}
    items = data.get("items", [])
    return {"count": len(items), "bytes": sum(int(i.get("bytes", 0)) for i in items),
            "local": sum(1 for i in items if i.get("origin") == "local"),
            "web": sum(1 for i in items if i.get("origin") == "web"),
            "video": sum(1 for i in items if i.get("origin") == "video"),
            "linked": sum(1 for i in items if i.get("linked"))}


def list_datasets(paths):
    root = datasets_root(paths)
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for slug in names:
        d = load_dataset(paths, slug)
        if d is None:
            continue
        st = dataset_stats(paths, slug, d)
        out.append({"slug": slug, "name": d.get("name", slug), "tag": d.get("tag", ""), "learn": d["learn"],
                    "created": d.get("created", ""), "path": _ds_dir(paths, slug), **st})
    return out


# -----------------------------------------------------------------------------
# Kolaj / bölünmüş ekran algılama: birden fazla kare yan yana ya da üst üste konmuş görseller.
# Model bu düzeni de öğrenir ve görselleri bölmeli çizer; eğitimden önce kullanıcıya sorulur.
# Yalnızca Pillow + numpy (ek model yok). Görüntünün ortasında tam boy, tek piksellik düz bir
# sınır arar; iki yanı farklı içerikse "dikiş" sayar. Tuğla/satranç gibi tekrar eden desenler elenir.
# -----------------------------------------------------------------------------
COLLAGE_SIDE = 320
COLLAGE_EDGE = 16          # komşu satırlar arası parlaklık farkı (0-255)
COLLAGE_SIDE_DIFF = 25     # dikişin iki yanındaki şeritlerin ortalama farkı: farklı kareler = yüksek
COLLAGE_MIN_COVER = 0.8    # dikiş görüntünün en az bu kadarında kesintisiz olmalı (kapı/çizgi kenarları elenir)
COLLAGE_MAX_SEAMS = 4      # bir yönde bundan fazla düz çizgi = desen (tuğla, ızgara), kolaj değil
EXCLUDED_DIR = "excluded"


def _collage_seams(a, lo=0.15, hi=0.85):
    """
    a: gri görüntü (H×W float). Dönüş: (düz çizgi sayısı, dikişler). Düz çizgi: tam genişlikte, tek piksellik
    keskin sınır. Dikiş: iki yanı da farklı içerik olan düz çizgi.
    """
    import numpy as np
    H, W = a.shape
    if H < 40 or W < 40:
        return 0, []
    d = np.zeros((H, W), np.float32)
    d[1:-1] = np.abs(a[2:] - a[:-2])
    cov = (d > COLLAGE_EDGE).mean(axis=1)
    lines, hits = [], []
    for y in range(max(9, int(H * lo)), min(H - 9, int(H * hi))):
        c = float(cov[y])
        if c < 0.5:
            continue
        nb = np.r_[cov[max(0, y - 12):max(0, y - 4)], cov[y + 5:y + 13]]
        if nb.size == 0 or c - float(nb.max()) < 0.3:
            continue
        lines.append(y)
        if c < COLLAGE_MIN_COVER:
            continue
        side = float(np.abs(a[y - 8:y - 2].mean(0) - a[y + 3:y + 9].mean(0)).mean())
        if side >= COLLAGE_SIDE_DIFF:
            hits.append(y)

    # Ayırıcı şerit: kareler arasında tek renk, ince, tam genişlikte bir bant (beyaz/siyah çerçeve).
    std = a.std(axis=1)
    flat = std < 6.0
    y, max_run = max(9, int(H * lo)), max(3, int(H * 0.06))
    while y < min(H - 9, int(H * hi)):
        if not flat[y]:
            y += 1
            continue
        r0 = y
        while y < H and flat[y]:
            y += 1
        r1 = y - 1
        if r1 - r0 + 1 > max_run or r0 < 12 or r1 > H - 13:
            continue
        above, below = a[r0 - 10:r0 - 1], a[r1 + 2:r1 + 11]
        if float(above.std(axis=1).mean()) < 15 or float(below.std(axis=1).mean()) < 15:
            continue
        band = float(a[r0:r1 + 1].mean())
        if abs(float(above[-3:].mean()) - band) < 12 and abs(float(below[:3].mean()) - band) < 12:
            continue    # şeridin iki yanı da aynı renk: ayırıcı değil, görüntünün bir parçası
        hits.append((r0 + r1) // 2)
        lines.append((r0 + r1) // 2)
    hits.sort()
    lines.sort()

    def group(ys):
        out = []
        for y in ys:
            if not out or y - out[-1] > 3:
                out.append(y)
        return out
    return len(group(lines)), group(hits)


def looks_like_collage(path):
    """Görsel kolaj / bölünmüş ekran gibi mi? Okunamazsa False (karar kullanıcıya bırakılır)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.draft("L", (COLLAGE_SIDE * 2, COLLAGE_SIDE * 2))
            g = im.convert("L")
        return _collage_img(g)
    except Exception:
        return False


def _collage_img(g):
    """looks_like_collage'ın görsel (PIL) üzerinde çalışan hâli."""
    try:
        import numpy as np
        from PIL import Image
        g = g.convert("L")
        s = float(COLLAGE_SIDE) / max(g.size)
        if s < 1:
            g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.BOX)
        a = np.asarray(g, dtype=np.float32)
        nh, sh = _collage_seams(a)
        nv, sv = _collage_seams(np.ascontiguousarray(a.T))
        if nh > COLLAGE_MAX_SEAMS or nv > COLLAGE_MAX_SEAMS:
            return False      # çok sayıda düz çizgi: tuğla, fayans, ızgara gibi bir desen
        return bool(sh or sv)
    except Exception:
        return False


def find_collage_items(paths, slug, progress_cb=None, abort_event=None):
    """
    Veri setindeki kolaj gibi görünen öğeler (kullanıcının 'kalsın' dediği öğeler hariç).
    Sonuç her öğeye işlenir ('collage'), böylece aynı görsel bir daha taranmaz.
    """
    data = load_dataset(paths, slug)
    if not data:
        return []
    items = data["items"]
    todo = [it for it in items if "collage" not in it]
    for i, it in enumerate(todo):
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
        it["collage"] = looks_like_collage(dataset_item_path(paths, slug, it))
        if progress_cb and (i % 10 == 0 or i + 1 == len(todo)):
            progress_cb(i + 1, len(todo))
    if todo:
        save_dataset(paths, slug, data)
    return [it for it in items if it.get("collage") and not it.get("collage_ok")]


def keep_dataset_items(paths, slug, files):
    """Kullanıcı 'kalsın' dedi: bu öğeler bir daha sorulmaz."""
    data = load_dataset(paths, slug)
    if not data:
        return 0
    files, n = set(files), 0
    for it in data["items"]:
        if it.get("file") in files:
            it["collage_ok"] = True
            n += 1
    save_dataset(paths, slug, data)
    return n


def exclude_dataset_items(paths, slug, files):
    """
    Öğeleri veri setinden çıkarır. Programın kopyaladığı dosyalar silinmez, veri setinin 'excluded'
    klasörüne taşınır; bağlantılı (kullanıcının kendi klasöründeki) dosyalara hiç dokunulmaz.
    """
    data = load_dataset(paths, slug)
    if not data:
        return 0
    files = set(files)
    keep, moved = [], 0
    ex_dir = os.path.join(_ds_dir(paths, slug), EXCLUDED_DIR)
    for it in data["items"]:
        if it.get("file") not in files:
            keep.append(it)
            continue
        moved += 1
        if it.get("linked") or os.path.isabs(it.get("file") or ""):
            continue
        src = dataset_item_path(paths, slug, it)
        if not os.path.exists(src):
            continue
        os.makedirs(ex_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(src))
        dst, k = os.path.join(ex_dir, base + ext), 2
        while os.path.exists(dst):
            dst, k = os.path.join(ex_dir, "%s_%d%s" % (base, k, ext)), k + 1
        try:
            shutil.move(src, dst)
        except OSError:
            pass
    data["items"] = keep
    save_dataset(paths, slug, data)
    return moved


def _trim_flat_edges(g, max_frac=0.03, flat=3.0):
    """Kesitin kenarlarındaki tek renkli ayırıcı şeritleri kırpar. g: gri numpy dizisi. Dönüş: (sol, üst, sağ, alt)."""
    h, w = g.shape
    top, bottom, left, right = 0, h, 0, w
    my, mx = max(1, int(h * max_frac)), max(1, int(w * max_frac))
    rs, cs = g.std(axis=1), g.std(axis=0)
    while top < my and rs[top] < flat:
        top += 1
    while h - bottom < my and bottom - 1 > top and rs[bottom - 1] < flat:
        bottom -= 1
    while left < mx and cs[left] < flat:
        left += 1
    while w - right < mx and right - 1 > left and cs[right - 1] < flat:
        right -= 1
    return left, top, right, bottom


def _line_candidates(a, min_cov=0.3, lo=0.15, hi=0.85):
    """Gevşek ölçüt: tam boy, düz ve tek piksellik sınır adayları (satır numaraları). a: gri dizi."""
    import numpy as np
    H, W = a.shape
    if H < 40 or W < 20:
        return []
    d = np.zeros((H, W), np.float32)
    d[1:-1] = np.abs(a[2:] - a[:-2])
    cov = (d > COLLAGE_EDGE * 0.75).mean(axis=1)
    out = []
    for y in range(max(9, int(H * lo)), min(H - 9, int(H * hi))):
        c = float(cov[y])
        if c < min_cov:
            continue
        nb = np.r_[cov[max(0, y - 12):max(0, y - 3)], cov[y + 4:y + 13]]
        if nb.size and c - float(nb.max()) >= 0.18 and c >= float(cov[y - 1:y + 2].max()):
            out.append(y)
    return out


def _complete_grid(a, boxes):
    """
    Satırlara (ya da sütunlara) bölünmüş bir kolajda her satırda AYNI yerde düz bir dikey sınır varsa orayı da keser.
    Seri çekim gibi birbirine çok benzeyen karelerde (2×2 ızgara) orta dikiş tek başına zayıf kalır; ama her satırda
    aynı x'te çıkması güçlü bir işarettir. boxes: küçük görüntü koordinatları (x0, y0, x1, y1).
    """
    import numpy as np
    H, W = a.shape
    for axis in ("rows", "cols"):
        if axis == "rows":
            if not all(b[0] <= 1 and b[2] >= W - 1 for b in boxes):
                continue
            subs = [a[b[1]:b[3], :].T for b in boxes]          # dikey sınırlar = devrik görüntüde satırlar
            length = W
        else:
            if not all(b[1] <= 1 and b[3] >= H - 1 for b in boxes):
                continue
            subs = [a[:, b[0]:b[2]] for b in boxes]
            length = H
        cand = [_line_candidates(np.ascontiguousarray(sb)) for sb in subs]
        if not cand or not all(cand):
            continue
        common = []
        for p in cand[0]:
            if all(any(abs(p - q) <= 2 for q in other) for other in cand[1:]):
                if not common or p - common[-1] > 3:
                    common.append(p)
        # ızgara düzeni: parçalar çok küçük olmasın, en fazla 3 kesim
        common = [p for p in common if length * 0.15 <= p <= length * 0.85][:3]
        if not common:
            continue
        cuts = [0] + common + [length]
        if min(b - a_ for a_, b in zip(cuts, cuts[1:])) < length * 0.15:
            continue
        out = []
        for b in boxes:
            for c0, c1 in zip(cuts, cuts[1:]):
                c0 = c0 + (1 if c0 else 0)
                if axis == "rows":
                    out.append((c0, b[1], c1, b[3]))
                else:
                    out.append((b[0], c0, b[2], c1))
        if axis == "rows":
            out.sort(key=lambda r: (r[1], r[0]))
        return out
    return _complete_partial(a, boxes)


def _complete_partial(a, boxes):
    """Yarım kalmış ızgara: bir satır (sütun) kesildi, komşusu kesilmedi. Komşuda aynı yerde düz bir sınır varsa keser."""
    import numpy as np
    xs = sorted({b[0] for b in boxes if b[0] > 1})
    ys = sorted({b[1] for b in boxes if b[1] > 1})
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        done = False
        for e in xs:
            if x0 + w * 0.15 <= e <= x1 - w * 0.15:
                cand = _line_candidates(np.ascontiguousarray(a[y0:y1, x0:x1].T), lo=0.1, hi=0.9)
                if any(abs((x0 + q) - e) <= 2 for q in cand):
                    q = min(cand, key=lambda q: abs((x0 + q) - e))
                    out += [(x0, y0, x0 + q, y1), (x0 + q + 1, y0, x1, y1)]
                    done = True
                    break
        if not done:
            for e in ys:
                if y0 + h * 0.15 <= e <= y1 - h * 0.15:
                    cand = _line_candidates(np.ascontiguousarray(a[y0:y1, x0:x1]), lo=0.1, hi=0.9)
                    if any(abs((y0 + q) - e) <= 2 for q in cand):
                        q = min(cand, key=lambda q: abs((y0 + q) - e))
                        out += [(x0, y0, x1, y0 + q), (x0, y0 + q + 1, x1, y1)]
                        done = True
                        break
        if not done:
            out.append(b)
    out.sort(key=lambda r: (r[1], r[0]))
    return out


def collage_boxes(img, min_side=256):
    """
    Kesilecek kareler (görselin kendi pikselleriyle). Önizleme, sayım ve kesim hep bunu kullanır, böylece gösterilen
    kutular ile gerçekte kesilenler aynıdır. Kare hâlâ kolaj gibiyse bir kez daha bölünür.
    """
    mp = max(64, int(min_side or 0))
    panels = collage_panels(img, min_px=mp)
    final = []
    for (x0, y0, x1, y1) in panels:
        sub = img.crop((x0, y0, x1, y1))
        inner = collage_panels(sub, min_px=mp) if _collage_img(sub) else []
        final += [(x0 + a, y0 + b, x0 + c, y0 + d) for a, b, c, d in inner] if inner else [(x0, y0, x1, y1)]
    # tek kare kaldıysa (öbür parça çok küçük olduğu için atıldı) bu bir kesim değildir
    return final if len(final) >= 2 else []


def collage_grid_boxes(img, rows, cols, min_side=0):
    """
    Elle bölme: görseli rows × cols eşit parçaya ayırır; her kesim ±%15 içinde en yakın düz sınıra (dikişe) oturtulur,
    ayırıcı şeritler kırpılır. Dönüş: kutular (asıl piksellerle), okuma sırasına göre.
    """
    import numpy as np
    from PIL import Image
    img = img.convert("RGB")
    W, H = img.size
    s = min(1.0, float(COLLAGE_SIDE) / max(W, H))
    small = img.convert("L")
    if s < 1:
        small = small.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BOX)
    a = np.asarray(small, dtype=np.float32)
    g_full = np.asarray(img.convert("L"), dtype=np.float32)

    def cuts(n, arr, length_full):
        L = arr.shape[0]
        cand = _line_candidates(np.ascontiguousarray(arr), min_cov=0.25, lo=0.05, hi=0.95) if n > 1 else []
        out = []
        for k in range(1, n):
            want = L * k / n
            near = [q for q in cand if abs(q - want) <= L * 0.15 / max(1, n - 1)]
            q = min(near, key=lambda q: abs(q - want)) if near else want
            out.append(int(round(q / s)))
        return [0] + out + [length_full]
    ys = cuts(rows, a, H)
    xs = cuts(cols, np.ascontiguousarray(a.T), W)
    boxes = []
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            l, t, r, b = _trim_flat_edges(g_full[y0:y1, x0:x1])
            bx = (x0 + l, y0 + t, x0 + r, y0 + b)
            if min(bx[2] - bx[0], bx[3] - bx[1]) >= max(16, int(min_side or 0) // 2):
                boxes.append(bx)
    return boxes


def collage_panels(img, min_px=128, max_depth=3):
    """
    Kolajı karelerine ayırır: [(sol, üst, sağ, alt), ...] — görselin kendi pikselleriyle, okuma sırasına göre.
    Düz dikişlerden ve tek renkli ayırıcı şeritlerden keser; iç içe düzenler (ör. solda büyük, sağda iki küçük)
    parça parça yeniden aranır. Ayırıcı şeritler kırpılır; çok küçük ve düz renk parçalar atılır.
    Ayrılamıyorsa boş liste.
    """
    import numpy as np
    from PIL import Image
    g_full = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = g_full.shape
    s = min(1.0, float(COLLAGE_SIDE) / max(H, W))
    small = img.convert("L")
    if s < 1:
        small = small.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BOX)
    a = np.asarray(small, dtype=np.float32)
    sh_, sw_ = a.shape

    def split(x0, y0, x1, y1, depth):
        sub = a[y0:y1, x0:x1]
        if depth >= max_depth or sub.shape[0] < 48 or sub.shape[1] < 48:
            return [(x0, y0, x1, y1)]
        nh, hs = _collage_seams(sub)
        nv, vs = _collage_seams(np.ascontiguousarray(sub.T))
        if nh > COLLAGE_MAX_SEAMS or nv > COLLAGE_MAX_SEAMS or not (hs or vs):
            return [(x0, y0, x1, y1)]
        out = []
        def pieces(seams, length):
            # dikiş satırı iki karenin karışımıdır: hiçbir kareye verilmez
            starts = [0] + [q + 1 for q in seams]
            ends = list(seams) + [length]
            return [(a_, b_) for a_, b_ in zip(starts, ends) if b_ - a_ >= 4]
        if vs:                                   # önce sütunlar, sonra her sütun kendi içinde
            for a_, b_ in pieces(vs, x1 - x0):
                out += split(x0 + a_, y0, x0 + b_, y1, depth + 1)
        else:
            for a_, b_ in pieces(hs, y1 - y0):
                out += split(x0, y0 + a_, x1, y0 + b_, depth + 1)
        return out

    boxes = split(0, 0, sw_, sh_, 0)
    if len(boxes) < 2:
        return []
    boxes = _complete_grid(a, boxes)
    panels = []
    for x0, y0, x1, y1 in boxes:
        X0, Y0 = int(round(x0 / s)), int(round(y0 / s))
        X1, Y1 = min(W, int(round(x1 / s))), min(H, int(round(y1 / s)))
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            continue
        crop = g_full[Y0:Y1, X0:X1]
        l, t, r, b = _trim_flat_edges(crop)
        X0, Y0, X1, Y1 = X0 + l, Y0 + t, X0 + r, Y0 + b
        if min(X1 - X0, Y1 - Y0) < min_px:
            continue
        if float(g_full[Y0:Y1, X0:X1].std()) < 3.0:        # düz renk (boşluk/çerçeve) parçası
            continue
        panels.append((X0, Y0, X1, Y1))
    return panels


def split_collage_items(paths, slug, files, min_side=256, progress_cb=None, abort_event=None, boxes=None):
    """
    Kolaj öğelerini karelerine böler: her kare veri setine ayrı görsel olarak eklenir (veri setinin adıyla), asıl
    görsel 'excluded' klasörüne taşınır (silinmez; bağlantılı dosyaya dokunulmaz).
    boxes: {dosya: [(x0, y0, x1, y1), ...]} — kullanıcının seçtiği kareler (önizlemede gösterilenlerin alınanları).
    Verilmezse kareler collage_boxes ile bulunur. Boş liste: hiçbir kare alınmaz, asıl görsel yalnızca çıkarılır.
    Dönüş: {"split": bölünen, "new": yeni görsel, "failed": [bölünemeyen dosyalar]}.
    """
    import io
    from PIL import Image
    data = load_dataset(paths, slug)
    if not data:
        return {"split": 0, "new": 0, "failed": list(files)}
    want = set(files)
    boxes = boxes or {}
    hashes = {i.get("sha1") for i in data["items"]}
    done, failed, new = [], [], 0
    todo = [it for it in list(data["items"]) if it.get("file") in want]
    for n, it in enumerate(todo):
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
        chosen = boxes.get(it.get("file"))
        try:
            with Image.open(dataset_item_path(paths, slug, it)) as im:
                im.load()
                fmt = (im.format or "PNG").upper()
                img = im.convert("RGBA" if (fmt == "PNG" and "A" in im.getbands()) else "RGB")
            if chosen is not None:
                panels = [tuple(int(v) for v in b) for b in chosen
                          if 0 <= b[0] < b[2] <= img.width and 0 <= b[1] < b[3] <= img.height]
            else:
                panels = collage_boxes(img, min_side)
        except Exception:
            panels = []
        if not panels and chosen is None:
            failed.append(it.get("file"))
            continue
        added = 0
        for (x0, y0, x1, y1) in panels:
            crop = img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            if crop.mode == "RGBA":
                crop.save(buf, "PNG", optimize=True)
                ofmt = "PNG"
            else:
                crop.save(buf, "JPEG", quality=95, subsampling=0)
                ofmt = "JPEG"
            meta = {"from_collage": it.get("file"), "collage": False}
            for k in ("license", "author", "title", "url"):
                if it.get(k):
                    meta[k] = it[k]
            if _add_item(paths, slug, data, hashes, buf.getvalue(), (ofmt, crop.width, crop.height),
                         it.get("origin") or "local", it.get("src") or it.get("file"), copy=True, meta=meta) == "added":
                added += 1
        new += added
        done.append(it.get("file"))
        if progress_cb:
            progress_cb(n + 1, len(todo))
    save_dataset(paths, slug, data)
    if done:
        exclude_dataset_items(paths, slug, done)
    return {"split": len(done), "new": new, "failed": failed}


def collage_preview_data(path, max_side=200, min_side=256):
    """Önizleme için: (küçük görsel, kutular (asıl piksellerle), ölçek). Kutular kesimde aynen kullanılır."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        img = im.convert("RGB")
    bx = collage_boxes(img, min_side)
    s = min(1.0, float(max_side) / max(img.size))
    th = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    return th, bx, s


def collage_preview(path, max_side=220, min_side=256):
    """Önizleme: küçük görsel + kesim çizgileri (kırmızı kutular). Kesimle aynı kareler kullanılır, böylece
    gösterilen kare sayısı gerçekte eklenecek sayıyla aynıdır. Dönüş: (PIL görsel, kare sayısı)."""
    from PIL import ImageDraw
    th, panels, s = collage_preview_data(path, max_side, min_side)
    d = ImageDraw.Draw(th)
    for x0, y0, x1, y1 in panels:
        d.rectangle([x0 * s, y0 * s, x1 * s - 1, y1 * s - 1], outline=(255, 60, 60), width=2)
    return th, len(panels)


def scan_image_files(folder, recursive=True, limit=200000):
    """Klasördeki görsel dosyaları (sıralı). Gizli/sistem klasörleri atlanır."""
    found = []
    if not os.path.isdir(folder):
        return found
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith((".", "$")))
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXTS):
                    found.append(os.path.join(root, f))
                    if len(found) >= limit:
                        return found
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(p):
                found.append(p)
    return found


VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts",
              ".3gp", ".flv")
VIDEO_FRAME_CHOICES = (30, 60, 120, 200, 400)
FRAME_MAX_SIDE = 1920          # 4K kareler 1920'ye küçültülür (eğitim 512–1024 px kullanır, disk boşa dolmasın)


def is_video_file(path):
    return path.lower().endswith(VIDEO_EXTS)


def scan_media_files(folder, recursive=True, limit=200000):
    """Klasördeki görseller ve videolar: (görseller, videolar). Gizli/sistem klasörleri atlanır."""
    images, videos = [], []
    if not os.path.isdir(folder):
        return images, videos
    walker = os.walk(folder) if recursive else [(folder, [], sorted(os.listdir(folder)))]
    for root, dirs, files in walker:
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "$")))
        for f in sorted(files):
            low = f.lower()
            p = os.path.join(root, f)
            if low.endswith(IMAGE_EXTS):
                images.append(p)
            elif low.endswith(VIDEO_EXTS):
                videos.append(p)
            else:
                continue
            if len(images) + len(videos) >= limit:
                return images, videos
    if not recursive:
        images = [p for p in images if os.path.isfile(p)]
        videos = [p for p in videos if os.path.isfile(p)]
    return images, videos


def split_media_files(files):
    files = list(files or [])
    return [f for f in files if f.lower().endswith(IMAGE_EXTS)], [f for f in files if is_video_file(f)]


def video_info(path):
    """{'fps', 'frames', 'duration', 'w', 'h'} ya da açılamazsa None (OpenCV gerekir)."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0.5 < fps < 1000):
            fps = 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n < 0 or n > 50_000_000:
            n = 0
        return {"fps": fps, "frames": n, "duration": n / fps if n else 0.0,
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    finally:
        cap.release()


FRAME_DARK, FRAME_BRIGHT, FRAME_FLAT, FRAME_BLUR = 18, 240, 10.0, 25.0
FRAME_SIMILAR = 7.0            # küçük gri önizlemelerin ortalama farkı bunun altındaysa "aynı sahne" sayılır


def frame_quality(frame):
    """(netlik, önizleme, sorun) — sorun: None | 'dark' | 'flat' | 'blur'. Netlik: Laplace varyansı (320 px'te)."""
    import cv2
    h, w = frame.shape[:2]
    s = 320.0 / max(h, w)
    small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype("float32")
    mean = float(gray.mean())
    if mean < FRAME_DARK or mean > FRAME_BRIGHT:
        return 0.0, thumb, "dark"
    if float(gray.std()) < FRAME_FLAT:
        return 0.0, thumb, "flat"
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharp, thumb, ("blur" if sharp < FRAME_BLUR else None)


def extract_video_frames(paths, slug, video_path, max_frames=120, min_side=256, progress_cb=None, abort_event=None):
    """
    Videodan veri setine kare ekler. Video eşit zaman dilimlerine bölünür; her dilimden birkaç aday kare okunur
    ve en NET olanı alınır. Kararmış/düz (geçiş) kareler, bulanık dilimler ve bir önceki kareyle neredeyse aynı
    sahneler atlanır. Kareler veri setinin adıyla kaydedilir (istanbul_sokak_0001.jpg).
    Dönüş: {'added', 'similar', 'unusable', 'dup', 'small', 'bad', 'bytes', 'seconds'}.
    """
    import cv2
    import numpy as np
    st = {"added": 0, "similar": 0, "unusable": 0, "dup": 0, "small": 0, "bad": 0, "bytes": 0, "seconds": 0.0}
    data = load_dataset(paths, slug)
    if data is None:
        raise RuntimeError("dataset not found: " + slug)
    hashes = {i.get("sha1") for i in data["items"]}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st["bad"] = 1
        return st
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0.5 < fps < 1000):
            fps = 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n < 0 or n > 50_000_000:
            n = 0
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if vw and vh and min(vw, vh) < int(min_side or 0):
            st["small"] = 1
            return st
        max_frames = max(1, int(max_frames))
        if n > 0:
            seg = max(1, n // max_frames)              # dilim uzunluğu (kare)
        else:
            seg = max(1, int(round(fps * 2)))          # süre bilinmiyorsa 2 saniyelik dilimler
        cand = max(1, seg // 4)                         # dilim başına ~4 aday
        cand = min(cand, max(1, int(round(fps / 2))))   # en az yarım saniyede bir aday
        recent = []                                     # son kaydedilen karelerin önizlemeleri
        best = None                                     # (netlik, kare, sıra, önizleme)
        idx = 0
        last_prog = 0.0

        def flush():
            nonlocal best
            if best is None:
                st["unusable"] += 1
                return
            sharp, frame, fidx, thumb = best
            best = None
            if any(float(np.abs(thumb - r).mean()) < FRAME_SIMILAR for r in recent):
                st["similar"] += 1
                return
            h, w = frame.shape[:2]
            s = FRAME_MAX_SIDE / float(max(h, w))
            if s < 1.0:
                frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            if min(h, w) < int(min_side or 0):
                st["small"] += 1
                return
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                st["bad"] += 1
                return
            raw = buf.tobytes()
            sec = round(fidx / fps, 1)
            res = _add_item(paths, slug, data, hashes, raw, ("JPEG", w, h), "video", f"{os.path.abspath(video_path)}#t={sec}",
                            copy=True, meta={"video": os.path.basename(video_path), "t": sec})
            if res == "added":
                st["added"] += 1
                st["bytes"] += len(raw)
                recent.append(thumb)
                del recent[:-3]
                if st["added"] % 50 == 0:
                    save_dataset(paths, slug, data)
            else:
                st["dup"] += 1

        def consider(frame, fidx):
            nonlocal best
            if frame is None:
                return
            sharp, thumb, problem = frame_quality(frame)
            if problem is None and (best is None or sharp > best[0]):
                best = (sharp, frame, fidx, thumb)

        # Uzun videoda (dilim > 8 sn) her kare çözülmez: her dilimin 3 noktasına atlanır. 1 saatlik 4K bir video
        # baştan sona çözülse dakikalar sürerdi. Atlama bu dosyada çalışmazsa baştan sıralı okumaya dönülür.
        seek_mode = n > 0 and seg > fps * 8
        if seek_mode:
            nseg = int(math.ceil(n / float(seg)))
            dead = 0
            for si in range(nseg):
                if st["added"] >= max_frames:
                    break
                if abort_event is not None and abort_event.is_set():
                    raise AbortedError("user")
                got = 0
                for q in (0.25, 0.5, 0.75):
                    pos = min(n - 1, int(si * seg + q * seg))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        got += 1
                        consider(frame, pos)
                dead = 0 if got else dead + 1
                if dead >= 3 and st["added"] == 0 and si < 6:
                    seek_mode = False              # bu dosyada atlama çalışmıyor → sıralı okuma
                    break
                flush()
                idx = min(n, (si + 1) * seg)
                if progress_cb and (idx / n - last_prog) >= 0.01:
                    last_prog = idx / n
                    progress_cb(idx, n)
            if not seek_mode:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                best, idx = None, 0
                st["unusable"] = 0
        if not seek_mode:
            while st["added"] < max_frames:
                if abort_event is not None and abort_event.is_set():
                    raise AbortedError("user")
                if not cap.grab():
                    break
                if idx % cand == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        consider(frame, idx)
                idx += 1
                if idx % seg == 0:
                    flush()
                    if progress_cb and n and (idx / n - last_prog) >= 0.01:
                        last_prog = idx / n
                        progress_cb(idx, n)
            if best is not None and st["added"] < max_frames:
                flush()
        st["seconds"] = (n if seek_mode else idx) / fps
    finally:
        cap.release()
        save_dataset(paths, slug, data)
    return st


def _image_info(data_or_path):
    """(format, genişlik, yükseklik) ya da geçersizse None. Görsel bombası korumasıyla açar."""
    import io
    from PIL import Image
    try:
        src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
        with Image.open(src) as im:
            fmt, size = im.format, im.size
            im.verify()
        if not fmt or size[0] < 1 or size[1] < 1 or size[0] * size[1] > 200_000_000:
            return None
        return fmt.upper(), size[0], size[1]
    except Exception:
        return None


_FMT_EXT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif"}


def dataset_file_base(data, slug):
    """Veri setine kopyalanan dosyaların ad kökü: veri setinin adı ('İstanbul Sokak' → 'istanbul_sokak')."""
    return ascii_name((data or {}).get("name") or slug, maxlen=32)


def _next_file_name(paths, slug, data, ext):
    """'istanbul_sokak_0001.jpg', '..._0002.jpg' … — sayaç dataset.json'da tutulur, var olan dosya ezilmez."""
    base = dataset_file_base(data, slug)
    d = os.path.join(_ds_dir(paths, slug), "images")
    no = int(data.get("next_no") or 0)
    if no <= 0:
        no = 1
        pat = re.compile(re.escape(base) + r"_(\d+)\.[a-z0-9]+$", re.I)
        for it in data.get("items", []):
            m = pat.search(it.get("file") or "")
            if m:
                no = max(no, int(m.group(1)) + 1)
    while True:
        stem = f"{base}_{no:04d}"
        if not any(os.path.exists(os.path.join(d, stem + e)) for e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
            break
        no += 1
    name = stem + ext
    data["next_no"] = no + 1
    return name


def _add_item(paths, slug, data, hashes, raw, info, origin, src, copy=True, meta=None):
    """Tek görseli veri setine ekler. Dönüş: 'added' | 'dup'. meta: lisans/atıf (internet) ya da video/saniye bilgisi.
    Kopyalanan dosya veri setinin adıyla adlandırılır (istanbul_sokak_0001.jpg)."""
    import hashlib
    h = hashlib.sha1(raw).hexdigest()
    if h in hashes:
        return "dup"
    fmt, w, hh = info
    item = {"sha1": h, "w": w, "h": hh, "bytes": len(raw), "origin": origin, "src": src,
            "added": time.strftime("%Y-%m-%d %H:%M")}
    if meta:
        item.update({k: v for k, v in meta.items() if v not in (None, "")})
    if copy:
        name = _next_file_name(paths, slug, data, _FMT_EXT.get(fmt, ".png"))
        dest = os.path.join(_ds_dir(paths, slug), "images", name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        check_space(dest, len(raw) / GB + 0.2)
        with open(dest + ".part", "wb") as f:
            f.write(raw)
        os.replace(dest + ".part", dest)
        item["file"] = "images/" + name
    else:
        item["file"] = src
        item["linked"] = True
    data["items"].append(item)
    hashes.add(h)
    return "added"


def import_images(paths, slug, files, copy=True, min_side=256, progress_cb=None, abort_event=None):
    """
    Bilgisayardaki görselleri veri setine ekler. Aynı görsel (içerik özeti aynı) iki kez eklenmez;
    bozuk ve küçük görseller atlanır. Dönüş: {"added", "dup", "small", "bad", "bytes"}.
    """
    data = load_dataset(paths, slug)
    if data is None:
        raise RuntimeError("dataset not found: " + slug)
    hashes = {i.get("sha1") for i in data["items"]}
    st = {"added": 0, "dup": 0, "small": 0, "bad": 0, "bytes": 0}
    total = len(files)
    try:
        for n, fpath in enumerate(files):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            try:
                if os.path.getsize(fpath) > 200 * MB:
                    st["bad"] += 1
                    continue
                with open(fpath, "rb") as f:
                    raw = f.read()
            except OSError:
                st["bad"] += 1
                continue
            info = _image_info(raw)
            if info is None:
                st["bad"] += 1
            elif min(info[1], info[2]) < int(min_side or 0):
                st["small"] += 1
            elif _add_item(paths, slug, data, hashes, raw, info, "local", os.path.abspath(fpath), copy) == "added":
                st["added"] += 1
                st["bytes"] += len(raw) if copy else 0
            else:
                st["dup"] += 1
            if progress_cb and (n % 5 == 0 or n + 1 == total):
                progress_cb(n + 1, total)
            if st["added"] and st["added"] % 200 == 0:
                save_dataset(paths, slug, data)        # uzun içe aktarmada ara kayıt
    finally:
        save_dataset(paths, slug, data)
    return st


def clean_url_list(text, limit=WEB_MAX_PER_BATCH):
    """Metinden http(s) adreslerini çıkarır; tekrarları atar, en fazla 'limit' adres döner."""
    out, seen = [], set()
    for tok in re.split(r"\s+", text or ""):
        tok = tok.strip().strip("<>\"'(),;")
        if re.match(r"^https?://[^\s/]+\.[^\s/]+", tok, re.I) and tok not in seen:
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                break
    return out


def download_images(paths, slug, urls, offline, min_side=256, progress_cb=None, abort_event=None, log_cb=None,
                    delay=WEB_MIN_DELAY, max_mb=WEB_MAX_MB, session=None, sleep=time.sleep):
    """
    Doğrudan görsel adreslerini SIRAYLA indirir (tek bağlantı, istekler arası en az 1 sn).
    Web sayfalarını taramaz, bağlantı izlemez. Üst üste 5 ağ hatasında durur (ağı yormamak için).
    urls: adres listesi ya da {"url": ..., lisans/atıf alanları} sözlükleri (Openverse sonuçları).
    """
    if offline:
        raise OfflineError("offline mode is on")
    import requests
    data = load_dataset(paths, slug)
    if data is None:
        raise RuntimeError("dataset not found: " + slug)
    urls = urls[:WEB_MAX_PER_BATCH]
    hashes = {i.get("sha1") for i in data["items"]}
    st = {"added": 0, "dup": 0, "small": 0, "bad": 0, "failed": 0, "bytes": 0, "stopped": False}
    own = session is None
    s = session or requests.Session()
    if own:
        s.headers["User-Agent"] = APP_UA
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    limit = int(max_mb * MB)
    fails = 0
    try:
        for n, entry in enumerate(urls):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            meta = dict(entry) if isinstance(entry, dict) else None
            url = meta.pop("url") if meta else entry
            if n:
                sleep(max(WEB_MIN_DELAY, float(delay)))
            try:
                with s.get(url, stream=True, timeout=(10, 30), allow_redirects=True) as r:
                    if r.status_code != 200:
                        raise ValueError(f"HTTP {r.status_code}")
                    ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                    if ctype and not ctype.startswith("image/") and ctype != "application/octet-stream":
                        raise ValueError("not an image: " + ctype)
                    if int(r.headers.get("content-length") or 0) > limit:
                        raise ValueError("too large")
                    buf = bytearray()
                    for chunk in r.iter_content(256 * 1024):
                        if abort_event is not None and abort_event.is_set():
                            raise AbortedError("user")
                        buf += chunk
                        if len(buf) > limit:
                            raise ValueError("too large")
                fails = 0
            except AbortedError:
                raise
            except ValueError as e:
                st["bad"] += 1
                if log_cb:
                    log_cb(f"   ✗ {url[:90]} — {e}")
                continue
            except Exception as e:
                st["failed"] += 1
                fails += 1
                if log_cb:
                    log_cb(f"   ✗ {url[:90]} — {type(e).__name__}")
                if fails >= 5:
                    st["stopped"] = True
                    break
                continue
            finally:
                if progress_cb:
                    progress_cb(n + 1, len(urls))
            raw = bytes(buf)
            info = _image_info(raw)
            if info is None:
                st["bad"] += 1
            elif min(info[1], info[2]) < int(min_side or 0):
                st["small"] += 1
            elif _add_item(paths, slug, data, hashes, raw, info, "web", url, True, meta=meta) == "added":
                st["added"] += 1
                st["bytes"] += len(raw)
            else:
                st["dup"] += 1
    finally:
        save_dataset(paths, slug, data)
        if own:
            s.close()
    return st


def _rate_available(headers):
    """Openverse'ün 'X-RateLimit-Available-*' başlıklarından en düşük kalan hakkı okur (yoksa None)."""
    vals = []
    for k, v in headers.items():
        if k.lower().startswith("x-ratelimit-available-"):
            try:
                vals.append(int(v))
            except (TypeError, ValueError):
                pass
    return min(vals) if vals else None


def openverse_search(query, max_results=40, licenses="cc0,pdm,by", category="all", min_side=0, session=None,
                     sleep=time.sleep, abort_event=None, api_url=None):
    """
    Openverse'te açık lisanslı görsel arar. Dönüş: (sonuçlar, bilgi).
    Her sonuç: {"url", "title", "creator", "license", "license_version", "license_url", "source_page",
               "provider", "attribution"} — indirilince veri setine lisans/atıf bilgisiyle kaydedilir.
    İstekler sırayla ve en az 1 sn arayla yapılır; kullanım sınırına (429) gelinince durur.
    """
    import requests
    api_url = api_url or OPENVERSE_API
    info = {"pages": 0, "rate_limited": False, "available": None, "total": 0}
    results, seen = [], set()
    own = session is None
    s = session or requests.Session()
    if own:
        s.headers["User-Agent"] = APP_UA
    pages_needed = min(OPENVERSE_MAX_PAGES, max(1, int(math.ceil(max_results / float(OPENVERSE_PAGE))) + 1))
    try:
        for page in range(1, pages_needed + 1):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            if page > 1:
                sleep(WEB_MIN_DELAY)
            params = {"q": query, "license": licenses, "page_size": OPENVERSE_PAGE, "page": page}
            if category in OPENVERSE_CATEGORIES and category != "all":
                params["category"] = category
            r = s.get(api_url, params=params, timeout=(10, 30))
            info["pages"] += 1
            avail = _rate_available(r.headers)
            if avail is not None:
                info["available"] = avail
            if r.status_code == 429:
                info["rate_limited"] = True
                break
            if r.status_code in (400, 404) and page > 1:     # sonuç derinliği sınırı / sayfa yok
                break
            r.raise_for_status()
            data = r.json()
            info["total"] = int(data.get("result_count") or 0)
            for it in data.get("results") or []:
                url = it.get("url")
                if not url or url in seen or it.get("mature"):
                    continue
                w, h = it.get("width") or 0, it.get("height") or 0
                if min_side and w and h and min(int(w), int(h)) < int(min_side):
                    continue
                seen.add(url)
                results.append({
                    "url": url, "title": (it.get("title") or "")[:200], "creator": (it.get("creator") or "")[:200],
                    "license": it.get("license") or "", "license_version": it.get("license_version") or "",
                    "license_url": it.get("license_url") or "", "source_page": it.get("foreign_landing_url") or "",
                    "provider": it.get("provider") or it.get("source") or "",
                    "attribution": (it.get("attribution") or "")[:500], "via": "openverse"})
                if len(results) >= max_results:
                    break
            if len(results) >= max_results or page >= int(data.get("page_count") or 0):
                break
    finally:
        if own:
            s.close()
    return results, info


def search_query_request(text):
    """LLM: kullanıcının (TR/EN/KO/DE) yazdığı konuyu kısa bir İngilizce görsel arama sorgusuna çevirir."""
    system = ("You convert a request into a short English search query for a database of openly licensed images "
              "(Creative Commons, public domain). Return ONLY a JSON object with the field \"query\": 2-6 English "
              "keywords. Keep proper names exactly as written. Do not add quotes or extra words.")
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    return system, "Request:\n" + (text or "").strip(), schema


def sanitize_search_query(d, fallback):
    q = re.sub(r"\s+", " ", str((d or {}).get("query", "")).replace('"', " ")).strip()
    return q[:120] if q else (fallback or "").strip()


# =============================================================================
# LoRA'lar — dışarıdan eklenenler (addons\) + programın ürettikleri (loras\) + stil kartları
#   * LoRA dosyası yüklenmeden önce yalnızca safetensors başlığı okunur (hangi modele ait olduğu anlaşılır).
#   * Yüklemek ve eğitmek için PEFT (Apache-2.0, ~1 MB) gerekir; yalnızca kullanıcı onayıyla kurulur.
# =============================================================================
PEFT_PIP = ["peft==0.22.0"]
PEFT_SIZE = "~1 MB"
LORA_MAX_FILES = 2000
BASE_LABELS = {"sd15": "SD 1.5", "sdxl": "SDXL", "sd2": "SD 2.x", "flux": "FLUX.1", "dit": "DiT", "any": "—",
               "unknown": "?", "flux2_klein_4b": "FLUX.2 4B", "flux2": "FLUX.2 (9B/dev)", "zimage_turbo": "Z-Image",
               "wan": "Wan (video)", "qwen_image": "Qwen-Image", "other": "?"}


def own_lora_dir(paths):
    return os.path.join(paths.root, "loras")


def peft_installed(paths):
    return os.path.exists(os.path.join(paths.site_packages(), "peft", "__init__.py"))


_PEFT_VERIFY = ("import json\ntry:\n    import peft\n    from diffusers.utils import USE_PEFT_BACKEND\n"
                "    print('AILS_PEFT ' + json.dumps({'ok': bool(USE_PEFT_BACKEND), 'ver': peft.__version__}))\n"
                "except BaseException as e:\n    print('AILS_PEFT ' + json.dumps({'ok': False, 'err': (type(e).__name__ + "
                "': ' + str(e))[:300]}))\n")


def verify_peft(paths, timeout=600):
    try:
        r = subprocess.run([paths.venv_python, "-c", _PEFT_VERIFY], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=sub_env(paths), **no_window_kwargs())
    except Exception as e:
        return {"ok": False, "err": str(e)}
    for line in (r.stdout or "").splitlines()[::-1]:
        if line.startswith("AILS_PEFT "):
            try:
                return json.loads(line[10:])
            except Exception:
                break
    return {"ok": False, "err": ((r.stderr or "") + (r.stdout or "")).strip()[-300:]}


def install_peft(paths, log_cb, msg, progress_cb=None, abort_event=None):
    """PEFT'i programın ortamına kurar. Önce '--no-deps' (torch'a dokunmaz); eksik bağımlılık varsa normal kurulum."""
    check_space(paths.root, 0.2)
    try:
        log_cb(msg("log_peft_install"))
        status, _out = run_pip(paths, ["--no-deps", "--only-binary=:all:"] + PEFT_PIP, log_cb, abort_event, progress_cb)
        raise_for_pip(status, paths, 0.2, _out)
        v = verify_peft(paths)
        if not v.get("ok"):
            status, _out = run_pip(paths, ["--only-binary=:all:"] + PEFT_PIP, log_cb, abort_event, progress_cb)
            raise_for_pip(status, paths, 0.5, _out)
            v = verify_peft(paths)
            if not v.get("ok"):
                raise RuntimeError("PEFT: " + str(v.get("err", "import failed")))
    finally:
        clean_dir(paths.tmp)
    importlib.invalidate_caches()
    return v


def peft_ready_in_process():
    """Bu süreçteki diffusers PEFT'i görüyor mu? diffusers, PEFT kurulmadan önce içe aktarıldıysa yeniden
    başlatma gerekir (bu bilgi diffusers içe aktarılırken bir kez belirlenir)."""
    if "diffusers" not in sys.modules:
        return True
    try:
        from diffusers.utils import USE_PEFT_BACKEND
        return bool(USE_PEFT_BACKEND)
    except Exception:
        return False


def read_safetensors_header(path, limit=64 * MB):
    """(metadata, {tensör adı: şekil}) — dosyanın yalnızca başı okunur, ağırlıklar yüklenmez."""
    import struct
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n <= 2 or n > limit:
            raise ValueError("bad safetensors header")
        hdr = json.loads(f.read(n).decode("utf-8"))
    meta = hdr.pop("__metadata__", None) or {}
    shapes = {k: v.get("shape") for k, v in hdr.items() if isinstance(v, dict)}
    return meta, shapes


# DiT tabanlı modellerin LoRA'ları anahtar adları ve boyutlarıyla birbirinden ayrılır. Yalnızca "DiT" demek yetmez:
# FLUX.1, FLUX.2, Z-Image, Wan (video), Qwen-Image… LoRA'ları birbirine uymaz.
FLUX2_KLEIN_4B = {"double": 5, "single": 20, "hidden": 3072}
ZIMAGE_ARCH = {"layers": 30, "refiner": 2, "dim": 3840}
_LORA_PREFIXES = ("base_model.model.", "transformer.", "diffusion_model.", "model.diffusion_model.", "unet.")


def _lora_module_names(shapes):
    """{modül adı: (giriş boyutu, çıkış boyutu)} — LoRA'nın dokunduğu katmanlar (önek ve lora sonekleri atılmış).
    kohya biçimi (lora_unet_double_blocks_0_img_attn_qkv) noktalı biçime çevrilir."""
    out = {}
    for k, shp in shapes.items():
        low = k.lower()
        m = re.match(r"(.+?)\.(lora_a|lora_b|lora_down|lora_up|lora\.down|lora\.up)(\.default)?\.weight$", low)
        if not m:
            continue
        name, part = m.group(1), m.group(2)
        if name.startswith("lora_unet_") or name.startswith("lora_transformer_"):
            name = name.split("_", 2)[2]
            name = re.sub(r"_(\d+)_", r".\1.", name)
            for a, b in (("double_blocks", "double_blocks"), ("single_blocks", "single_blocks")):
                name = name.replace(a + "_", a + ".")
            name = re.sub(r"(img|txt)_(attn|mlp|mod)_", r"\1_\2.", name)
        for pre in _LORA_PREFIXES:
            if name.startswith(pre):
                name = name[len(pre):]
                break
        cur = list(out.get(name, (None, None)))
        if shp and len(shp) >= 2:
            if part in ("lora_a", "lora_down", "lora.down"):
                cur[0] = int(shp[1])
            else:
                cur[1] = int(shp[0])
        out[name] = tuple(cur)
    return out


def _max_index(names, block):
    idx = [int(m.group(1)) for n in names for m in [re.match(r"(?:.*\.)?" + block + r"\.(\d+)\.", n + ".")] if m]
    return max(idx) if idx else -1


def detect_dit_lora(shapes):
    """DiT LoRA'sının ait olduğu model: flux2_klein_4b | flux2 | flux | zimage_turbo | wan | qwen_image | None."""
    mods = _lora_module_names(shapes)
    if not mods:
        return None
    names = list(mods)
    joined = " ".join(names)
    if any(t in joined for t in ("down_blocks", "up_blocks", "mid_block", "input_blocks", "output_blocks",
                                 "middle_block")):
        return None              # SD 1.5 / SDXL UNet LoRA'sı
    # Wan (video) ve Qwen-Image: kendi adlandırmaları
    if re.search(r"(^|\s)blocks\.\d+\.(cross_attn|self_attn|ffn)\.", " " + joined):
        return "wan"
    if "img_mlp.net" in joined or "txt_mlp.net" in joined or ".img_mod.1" in joined:
        return "qwen_image"
    # Z-Image: layers.N.attention / feed_forward, noise_refiner, context_refiner
    if re.search(r"(^|\s)(layers|noise_refiner|context_refiner)\.\d+\.(attention|feed_forward|adaln_modulation)",
                 " " + joined):
        dims = {d for v in mods.values() for d in v if d}
        if _max_index(names, "layers") < ZIMAGE_ARCH["layers"] and ZIMAGE_ARCH["dim"] in dims:
            return "zimage_turbo"
        return "other"
    is_flux = any(t in joined for t in ("double_blocks", "single_blocks", "transformer_blocks", "single_transformer_blocks"))
    if not is_flux:
        return None
    flux2_marks = ("to_qkv_mlp_proj", "ff.linear_in", "ff_context.linear_in", "double_stream_modulation",
                   "single_stream_modulation")
    flux1_marks = ("proj_mlp", "ff.net.", "ff_context.net.", "norm1.linear", "norm.linear")
    out_dims = {v[1] for v in mods.values() if v[1]}
    in_dims = {v[0] for v in mods.values() if v[0]}
    dbl = max(_max_index(names, "double_blocks"), _max_index(names, "transformer_blocks"))
    sgl = max(_max_index(names, "single_blocks"), _max_index(names, "single_transformer_blocks"))
    if any(t in joined for t in flux1_marks) or out_dims & {21504, 12288}:
        return "flux"
    if any(t in joined for t in flux2_marks) or out_dims & {27648, 18432}:
        k = FLUX2_KLEIN_4B
        if dbl < k["double"] and sgl < k["single"] and (not in_dims or k["hidden"] in in_dims):
            return "flux2_klein_4b"
        return "flux2"
    # adlandırmadan ayrılamadı: blok sayısına bak (FLUX.1: 19 + 38 blok)
    if dbl >= FLUX2_KLEIN_4B["double"] or sgl >= FLUX2_KLEIN_4B["single"]:
        return "flux"
    return "other"


def detect_lora_base(meta, shapes):
    """LoRA'nın hangi temel modele ait olduğu: sd15 | sdxl | sd2 | flux | flux2_klein_4b | flux2 | zimage_turbo |
    wan | qwen_image | other | unknown."""
    own = str(meta.get("ails_base", ""))
    if own in IMAGE_MODEL_BY_KEY:
        return own
    dit = detect_dit_lora(shapes)
    if dit:
        return dit
    mv = str(meta.get("ss_base_model_version", "")).lower()
    arch = str(meta.get("modelspec.architecture", "")).lower()
    if mv.startswith("sdxl") or "stable-diffusion-xl" in arch:
        return "sdxl"
    if mv.startswith("sd_v1") or "stable-diffusion-v1" in arch:
        return "sd15"
    if mv.startswith("sd_v2") or "stable-diffusion-v2" in arch:
        return "sd2"
    low = {k.lower(): v for k, v in shapes.items()}
    if any(k.startswith(("lora_te2_", "te2.", "text_encoder_2.")) for k in low):
        return "sdxl"
    for k, shp in low.items():
        # çapraz dikkat (cross-attention) to_k giriş boyutu: 768 = SD 1.x, 1024 = SD 2.x, 2048 = SDXL
        if "attn2" in k and "to_k" in k and ("lora_down" in k or "lora_a" in k or "lora.down" in k) and shp and len(shp) >= 2:
            return {768: "sd15", 1024: "sd2", 2048: "sdxl"}.get(int(shp[-1]), "unknown")
    if any("input_blocks" in k for k in low):
        return "sdxl"
    if "flux" in mv or "flux" in arch:
        return "flux"
    if any(k.startswith(("transformer.", "diffusion_model.")) for k in low):
        return "other"
    return "unknown"


def lora_compat(base, model_key):
    """'yes' | 'no' | 'maybe' — LoRA seçili görsel modeliyle (mimarisiyle) çalışır mı?"""
    if base == "any":
        return "yes"
    if base in ("sd15", "sdxl", "flux2_klein_4b", "zimage_turbo"):
        return "yes" if base == model_key else "no"
    if base in ("sd2", "flux", "flux2", "wan", "qwen_image", "other", "dit"):
        return "no"
    # tanınmayan biçim: SD 1.5 / SDXL'de belki (eski kohya biçimleri), DiT modellerinde hayır
    return "maybe" if model_key in ("sd15", "sdxl") else "no"


def _top_tags(meta, n=4):
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
        counts = {}
        for d in freq.values():
            for tag, c in d.items():
                counts[tag.strip()] = counts.get(tag.strip(), 0) + int(c)
        return ", ".join(t for t, _c in sorted(counts.items(), key=lambda x: -x[1])[:n])
    except Exception:
        return ""


# =============================================================================
# Kullanıcının eklediği ana modeller (checkpoint) — addons\models\ (ya da yanlışlıkla addons\ içine konanlar)
# Program bunları İNDİRMEZ; yalnızca kullanıcının koyduğu .safetensors dosyalarını okur. .ckpt/.pt (pickle,
# içinde kod çalıştırabilir) hiç açılmaz. Desteklenen mimariler: SD 1.5 ve SDXL.
# =============================================================================
CUSTOM_MODELS_SUBDIR = "models"
CUSTOM_PREFIX = "custom:"
CUSTOM_FAMILIES = ("sd15", "sdxl")
UNSAFE_MODEL_EXTS = (".ckpt", ".pt", ".pth", ".bin")
_LORA_MARKERS = ("lora_down", "lora_up", "lora_a.", "lora_b.", ".lora.", "lora_te", "lora_unet", "hada_w",
                 "lokr_w", ".oft_", "lora_mid")
_custom_cache = {}     # yol -> (boyut, değişiklik zamanı, sınıflandırma)


def custom_models_dir(paths):
    return os.path.join(addons_dir(paths), CUSTOM_MODELS_SUBDIR)


def classify_safetensors(meta, shapes):
    """
    (tür, ayrıntı) — tür: 'lora' | 'sd15' | 'sdxl' | 'unsupported' | 'unknown'.
    'unsupported' için ayrıntı: 'sd2' | 'refiner' | 'inpaint' | 'flux' | 'sd3' | 'other' | 'novae'.
    """
    keys = [k.lower() for k in shapes]
    if any(m in k for k in keys for m in _LORA_MARKERS):
        return "lora", ""
    has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    if not has_unet:
        if any(("double_blocks" in k or "single_blocks" in k) for k in keys):
            return "unsupported", "flux"
        if any(k.startswith(("transformer.", "diffusion_model.", "model.diffusion_model.joint_blocks")) for k in keys):
            return "unsupported", "other"
        return "unknown", ""
    low = {k.lower(): v for k, v in shapes.items()}
    if any("double_blocks" in k or "single_blocks" in k for k in keys):
        return "unsupported", "flux"
    if any("joint_blocks" in k for k in keys):
        return "unsupported", "sd3"
    conv_in = low.get("model.diffusion_model.input_blocks.0.0.weight") or []
    if len(conv_in) >= 2 and int(conv_in[1]) != 4:
        return "unsupported", "inpaint"
    if not any(k.startswith(("first_stage_model.", "vae.")) for k in keys):
        return "unsupported", "novae"
    if any(k.startswith("conditioner.embedders.1.") for k in keys):
        return "sdxl", ""
    if any(k.startswith("conditioner.embedders.0.") for k in keys):
        return "unsupported", "refiner"
    if any(k.startswith("cond_stage_model.model.") for k in keys):
        return "unsupported", "sd2"
    if any(k.startswith("cond_stage_model.transformer.") for k in keys):
        for k, shp in low.items():
            if "attn2.to_k.weight" in k and shp and len(shp) == 2:
                return ("unsupported", "sd2") if int(shp[1]) == 1024 else ("sd15", "")
        return "sd15", ""
    return "unknown", ""


def _classify_file(path):
    try:
        st = os.stat(path)
    except OSError:
        return "unknown", ""
    sig = (st.st_size, int(st.st_mtime))
    hit = _custom_cache.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    try:
        meta, shapes = read_safetensors_header(path)
        res = classify_safetensors(meta, shapes)
    except Exception:
        res = ("unknown", "broken")
    _custom_cache[path] = (sig, res)
    return res


def scan_custom_models(paths, limit=200):
    """
    addons\\models\\ (alt klasörler dahil) ve addons\\ kökündeki tam modeller.
    Dönüş: (modeller, atlananlar) — atlananlar: [(dosya adı, neden)], neden: unsafe | sd2 | flux | … | unknown.
    """
    found, skipped = [], []
    seen = set()
    cands = []
    md = custom_models_dir(paths)
    if os.path.isdir(md):
        for root, dirs, files in os.walk(md):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            cands += [os.path.join(root, f) for f in sorted(files)]
    ad = addons_dir(paths)
    if os.path.isdir(ad):
        cands += [os.path.join(ad, f) for f in sorted(os.listdir(ad)) if f.lower().endswith(".safetensors")]
    for p in cands[:limit * 4]:
        low = p.lower()
        in_models = os.path.normpath(os.path.dirname(p)).startswith(os.path.normpath(md))
        if low.endswith(UNSAFE_MODEL_EXTS) and in_models:
            skipped.append((os.path.basename(p), "unsafe"))
            continue
        if not low.endswith(".safetensors") or not os.path.isfile(p):
            continue
        kind, why = _classify_file(p)
        if kind in CUSTOM_FAMILIES:
            rel = os.path.relpath(p, ad).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            found.append(_custom_model_entry(paths, p, rel, kind))
        elif in_models or kind == "unsupported":
            skipped.append((os.path.basename(p), why or kind))
        if len(found) >= limit:
            break
    return found, skipped


def _custom_model_entry(paths, path, rel, family):
    base = IMAGE_MODEL_BY_KEY[family]
    m = dict(base)
    m.update({
        "key": CUSTOM_PREFIX + rel, "name": os.path.splitext(os.path.basename(path))[0], "family": family,
        "custom": True, "file": path, "repo": None, "variant": None, "license": "—", "desc": "desc_custom",
        "size_gb": round(os.path.getsize(path) / GB, 2), "models_root": paths.models,
        "misplaced": os.path.normpath(os.path.dirname(path)) == os.path.normpath(addons_dir(paths)),
        "load_kwargs": dict(base.get("load_kwargs", {})),
    })
    return m


def refresh_custom_models(paths):
    """Kullanıcı modellerini IMAGE_MODEL_BY_KEY'e işler (eskiler çıkarılır). Dönüş: (modeller, atlananlar)."""
    found, skipped = scan_custom_models(paths)
    for k in [k for k in IMAGE_MODEL_BY_KEY if k.startswith(CUSTOM_PREFIX)]:
        del IMAGE_MODEL_BY_KEY[k]
    for m in found:
        IMAGE_MODEL_BY_KEY[m["key"]] = m
    return found, skipped


def custom_config_dir(models_root, family):
    """
    Tek dosyalık modeli açmak için gereken ayar dosyaları (model ağırlığı değil): aynı mimarideki programın modeli
    kuruluysa onun klasörü, değilse küçük ayar paketi (models\\configs\\<mimari>). İkisi de yoksa None.
    """
    base = IMAGE_MODEL_BY_KEY[family]
    if is_image_model_installed(models_root, base):
        return image_model_dir(models_root, base)
    d = os.path.join(models_root, "configs", family)
    return d if os.path.exists(os.path.join(d, MARKER)) else None


CONFIG_PACK_SIZE = "~4 MB"


def download_model_config(models_root, family, progress_cb=None, abort_event=None):
    """Yalnızca ayar dosyaları (json/txt: yapı, zamanlayıcı, sözlük) — birkaç MB, ağırlık yok."""
    from huggingface_hub import snapshot_download
    base = IMAGE_MODEL_BY_KEY[family]
    dest = os.path.join(models_root, "configs", family)
    os.makedirs(dest, exist_ok=True)
    result = {"err": None}

    def _run():
        try:
            snapshot_download(base["repo"], local_dir=dest, allow_patterns=["model_index.json", "*/*.json", "*/*.txt"],
                              ignore_patterns=["*.safetensors", "*.bin", "*.ckpt", "*.onnx*", "*.msgpack", "*.h5"],
                              max_workers=2)
        except Exception as e:  # noqa
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.5)
        if progress_cb:
            progress_cb(folder_size_bytes(dest), 4 * MB)
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
    if result["err"] is not None:
        raise result["err"]
    if not os.path.exists(os.path.join(dest, "model_index.json")):
        raise RuntimeError("model_index.json missing — config download incomplete")
    with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
        json.dump({"repo": base["repo"], "config_only": True, "date": time.strftime("%Y-%m-%d %H:%M")}, f)
    return dest


class CustomModelConfigMissing(RuntimeError):
    pass


def load_custom_pipeline(model, kwargs):
    """Tek dosyalık SD 1.5 / SDXL modelini ağ kullanmadan açar (ayarlar yerel klasörden)."""
    import diffusers
    family = model["family"]
    cfg = custom_config_dir(model["models_root"], family)
    if not cfg:
        raise CustomModelConfigMissing(family)
    cls = diffusers.StableDiffusionXLPipeline if family == "sdxl" else diffusers.StableDiffusionPipeline
    return cls.from_single_file(model["file"], config=cfg, local_files_only=True, **kwargs)


def scan_loras(paths):
    """Tüm LoRA'lar ve stil kartları. Kendi ürettiklerimizin yanında aynı adlı .json bilgi dosyası bulunur."""
    out = []
    for origin, d in (("own", own_lora_dir(paths)), ("external", addons_dir(paths))):
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = sorted(x for x in dirs if not x.endswith("_samples") and not x.startswith(".") and
                             not (origin == "external" and os.path.normpath(root) == os.path.normpath(d) and
                                  x.lower() == CUSTOM_MODELS_SUBDIR))
            for fn in sorted(files):
                if len(out) >= LORA_MAX_FILES:
                    return out
                p = os.path.join(root, fn)
                low = fn.lower()
                if low.endswith(".card.json") and origin == "own":
                    try:
                        with open(p, encoding="utf-8") as f:
                            card = json.load(f)
                    except Exception:
                        continue
                    out.append({"id": "card:" + fn, "kind": "card", "origin": "own", "name": card.get("name", fn),
                                "path": p, "base": "any", "size_mb": 0.0, "trigger": "", "tag": card.get("tag", ""),
                                "style": card.get("style", ""), "avoid": card.get("avoid", ""),
                                "type": "card", "learn": norm_learn(card.get("learn")),
                                "created": card.get("created", ""), "sample": ""})
                elif low.endswith(".safetensors"):
                    info = {}
                    side = os.path.splitext(p)[0] + ".json"
                    if os.path.exists(side):
                        try:
                            with open(side, encoding="utf-8") as f:
                                info = json.load(f)
                        except Exception:
                            info = {}
                    try:
                        meta, shapes = read_safetensors_header(p)
                        if origin == "external" and classify_safetensors(meta, shapes)[0] in \
                                CUSTOM_FAMILIES + ("unsupported",):
                            continue      # tam model (checkpoint) LoRA değildir: model listesinde görünür
                        base = info.get("base") or detect_lora_base(meta, shapes)
                    except Exception:
                        meta, base = {}, "unknown"
                    typ = info.get("type", "")
                    learn = info.get("learn") or (typ if typ in LEARN_KINDS else "")
                    out.append({"id": f"{origin}:" + os.path.relpath(p, d).replace("\\", "/"), "kind": "lora",
                                "origin": origin, "name": info.get("name") or os.path.splitext(fn)[0], "path": p,
                                "base": base, "size_mb": os.path.getsize(p) / MB, "trigger": info.get("trigger", ""),
                                "tag": info.get("tag") or _top_tags(meta), "type": typ, "learn": learn,
                                "created": info.get("created", ""), "sample": info.get("sample", "")})
    return out


def trash_dir(paths):
    """Silinen LoRA/kartların gittiği yer (tarama bu klasörü görmez)."""
    return os.path.join(own_lora_dir(paths), ".trash")


def trash_own_item(paths, item):
    """Programın ürettiği bir LoRA'yı ya da kelime kartını kalıcı silmez: loras\\.trash\\<tarih-saat>\\ içine taşır
    (LoRA ise bilgi dosyası ve örnek görselleriyle). Dönüş: (klasör, taşınanlar)."""
    d = os.path.join(trash_dir(paths), time.strftime("%Y%m%d-%H%M%S"))
    n = 2
    while os.path.exists(d):
        d = os.path.join(trash_dir(paths), time.strftime("%Y%m%d-%H%M%S") + f"_{n}")
        n += 1
    os.makedirs(d)
    path = item["path"]
    if item.get("kind") == "card":
        files = [path]
    else:
        base = os.path.splitext(path)[0]
        files = [path, base + ".json", base + "_samples"]
    moved = []
    for f in files:
        if os.path.exists(f):
            dst = os.path.join(d, os.path.basename(f))
            shutil.move(f, dst)
            moved.append(dst)
    return d, moved


def find_lora(items, lora_id):
    for it in items:
        if it["id"] == lora_id:
            return it
    return None


def lora_trigger_text(item):
    """Seçilen LoRA'nın prompt'a eklenecek tetikleyici kelimeleri (yalnızca programın ürettikleri için)."""
    # "network": ağdaki sunucunun LoRA'sı — sunucu tetikleyiciyi yalnızca kendi ürettikleri için gönderir
    return (item.get("trigger") or "").strip() if item.get("origin") in ("own", "network") else ""


# =============================================================================
# Görselleri otomatik okuma (açıklama yazma) — Florence-2 (Microsoft, MIT), transformers'ın kendi sürümü
# =============================================================================
CAPTIONER = {"key": "florence2_base_ft", "name": "Florence-2 base (Microsoft)", "repo": "florence-community/Florence-2-base-ft",
             "size": "~0.5–0.9 GB", "size_gb": 0.9, "license": "MIT", "desc": "desc_captioner"}
CAPTION_TASK = "<DETAILED_CAPTION>"


def captioner_dir(models_root):
    return os.path.join(models_root, "caption", CAPTIONER["key"])


def is_captioner_installed(models_root):
    return os.path.exists(os.path.join(captioner_dir(models_root), MARKER))


def download_captioner(models_root, progress_cb, abort_event):
    from huggingface_hub import snapshot_download
    dest = captioner_dir(models_root)
    allow = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
    total = int(CAPTIONER["size_gb"] * GB)
    try:
        infos = snapshot_download(CAPTIONER["repo"], allow_patterns=allow, dry_run=True)
        total = sum((i.file_size or 0) for i in infos) or total
    except Exception:
        pass
    already = folder_size_bytes(dest) if os.path.isdir(dest) else 0
    check_space(models_root, max(0, total - already) / GB + 0.5)
    os.makedirs(dest, exist_ok=True)
    result = {"err": None}

    def _run():
        try:
            snapshot_download(CAPTIONER["repo"], local_dir=dest, allow_patterns=allow, max_workers=2)
        except Exception as e:  # noqa
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(1.0)
        if progress_cb:
            progress_cb(folder_size_bytes(dest), total)
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
    if result["err"] is not None:
        err = result["err"]
        if isinstance(err, OSError) and getattr(err, "errno", None) == 28:
            raise DiskFullError(drive_label(models_root), total / GB, disk_free_gb(models_root))
        raise err
    if not os.path.exists(os.path.join(dest, "config.json")):
        raise RuntimeError("config.json missing — download incomplete")
    with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
        json.dump({"repo": CAPTIONER["repo"], "size_gb": round(folder_size_bytes(dest) / GB, 2),
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f)
    return dest


_CAP_PREFIX = re.compile(r"^(the image (shows|depicts|features|is)|this image (shows|depicts|is)|in this image,?|"
                         r"the picture (shows|depicts)|an image of)\s+", re.I)


def clean_caption(text):
    t = re.sub(r"</?s>|<pad>", "", text or "").strip()
    t = _CAP_PREFIX.sub("", t).strip()
    return (t[:1].upper() + t[1:])[:600] if t else ""


class Captioner:
    """Florence-2 ile görsel açıklaması. GPU varsa fp16, yoksa CPU fp32. İş bitince close() çağrılmalı."""

    def __init__(self, path, device="cpu"):
        torch = import_torch()
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Florence2ForConditionalGeneration.from_pretrained(path).to(device=device, dtype=self.dtype).eval()

    def caption(self, img, task=CAPTION_TASK, max_new_tokens=160):
        torch = self.torch
        img = img.convert("RGB")
        inputs = self.processor(text=task, images=img, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        try:
            parsed = self.processor.post_process_generation(text, task=task, image_size=(img.width, img.height))
            text = parsed.get(task, text) if isinstance(parsed, dict) else text
        except Exception:
            pass
        return clean_caption(text if isinstance(text, str) else str(text))

    def close(self):
        self.model = None
        self.processor = None
        free_gpu_memory()


def dataset_item_path(paths, slug, item):
    f = item.get("file") or ""
    return f if (item.get("linked") or os.path.isabs(f)) else os.path.join(_ds_dir(paths, slug), f)


def caption_dataset(paths, slug, caption_fn, overwrite=False, progress_cb=None, abort_event=None):
    """Açıklaması olmayan görselleri okur ve dataset.json'a yazar. caption_fn(PIL) -> str."""
    from PIL import Image
    data = load_dataset(paths, slug)
    if data is None:
        raise RuntimeError("dataset not found: " + slug)
    todo = [it for it in data["items"] if overwrite or not it.get("caption")]
    st = {"done": 0, "failed": 0, "total": len(todo)}
    try:
        for n, it in enumerate(todo):
            if abort_event is not None and abort_event.is_set():
                raise AbortedError("user")
            try:
                with Image.open(dataset_item_path(paths, slug, it)) as im:
                    im.load()
                    cap = caption_fn(im)
                if cap:
                    it["caption"] = cap
                    st["done"] += 1
                else:
                    st["failed"] += 1
            except AbortedError:
                raise
            except Exception:
                st["failed"] += 1
            if progress_cb:
                progress_cb(n + 1, len(todo))
            if st["done"] and st["done"] % 20 == 0:
                save_dataset(paths, slug, data)
    finally:
        save_dataset(paths, slug, data)
    return st


# =============================================================================
# Stil kartı (hafif mod): görsellerin ORTAK tarzını kelimelere döker — kimlik/kişi değil, tarz
# =============================================================================
STYLE_WORDS = [
    "watercolor", "oil painting", "acrylic", "gouache", "pastel", "charcoal", "pencil", "sketch", "ink", "line art",
    "lineart", "manga", "anime", "comic", "cartoon", "cel shading", "pixel art", "3d render", "low poly", "vector",
    "flat colors", "minimalist", "impressionist", "ukiyo-e", "woodblock", "art nouveau", "baroque", "surreal",
    "abstract", "photograph", "photo", "film grain", "black and white", "monochrome", "sepia", "vintage", "retro",
    "neon", "cyberpunk", "fantasy", "dark", "moody", "bright", "vibrant", "colorful", "muted colors", "pastel colors",
    "warm tones", "cool tones", "soft lighting", "dramatic lighting", "backlight", "sunset", "golden hour",
    "high contrast", "bokeh", "close-up", "portrait", "landscape", "cinematic", "detailed", "textured", "brush strokes",
    "halftone", "screentone", "chibi", "realistic", "illustration", "digital painting", "concept art", "glitch",
]


PLACE_WORDS = [
    "street", "narrow street", "alley", "cobblestone", "sidewalk", "road", "avenue", "square", "plaza", "market",
    "bazaar", "shop", "storefront", "cafe", "restaurant", "street vendor", "sign", "building", "old building",
    "apartment building", "balcony", "facade", "brick", "stone", "wooden house", "architecture", "ottoman", "mosque",
    "minaret", "dome", "church", "tower", "castle", "bridge", "waterfront", "sea", "river", "harbor", "pier", "ferry",
    "boat", "seagull", "tram", "bus", "taxi", "car", "traffic", "tree", "park",
    "hill", "stairs", "graffiti", "street lamp", "lights", "night", "evening", "sunset", "daytime", "sunny", "cloudy",
    "rain", "wet", "snow", "skyline", "city", "cityscape", "downtown", "neighborhood", "historic", "modern", "urban",
    "village", "countryside", "mountain", "beach", "garden", "fountain", "statue", "flag",
]
ANATOMY_WORDS = [
    "full body", "upper body", "standing", "sitting", "kneeling", "lying", "walking", "running", "jumping", "dancing",
    "stretching", "crouching", "leaning", "pose", "dynamic pose", "front view", "side view", "back view",
    "three-quarter view", "from above", "from below", "hands", "hand", "fingers", "arms", "legs", "feet", "torso",
    "shoulders", "neck", "muscular", "athletic", "slim", "proportions", "anatomy", "hands on hips", "arms crossed",
    "arms raised", "holding", "reaching", "pointing", "gesture", "figure", "figure drawing", "life drawing",
    "mannequin", "plain background", "white background", "studio",
]
_CARD_STOP = set("""a an the of in on at to by for from with without and or but is are was were be been being it its
this that these those there here which who whom whose what when where while as into onto over under above below near
next behind front back left right top bottom middle center side sides image picture photo photograph shows showing
show depicts depicting features featuring appears appear seems looks look visible also very some several many other
another one two three has have having can could may might would will just only each both all any more most such than
then their them they he she his her him we our you your i me my out up down off about around through between among
background foreground scene view close""".split())
FACE_WORDS = [
    "long hair", "short hair", "curly hair", "straight hair", "wavy hair", "bangs", "ponytail", "braid", "bun",
    "blonde hair", "brown hair", "black hair", "red hair", "gray hair", "white hair", "bald", "beard", "mustache",
    "glasses", "sunglasses", "freckles", "mole", "dimples", "smile", "smiling", "serious", "blue eyes", "brown eyes",
    "green eyes", "makeup", "lipstick", "earrings", "round face", "oval face", "sharp jawline", "high cheekbones",
    "thick eyebrows", "young", "elderly", "close-up", "portrait",
]
ANATOMY_AVOID = ("bad anatomy, bad proportions, extra fingers, missing fingers, fused fingers, extra limbs, "
                 "mutated hands, twisted limbs")
_CARD_VOCAB = {"style": STYLE_WORDS, "place": PLACE_WORDS, "anatomy": ANATOMY_WORDS, "subject": None,
               "face": FACE_WORDS}
_PEOPLE_TERMS = re.compile(r"^(people|persons?|walking|standing|sitting|crowds?|pedestrians?|tourists?|wearing|shirts?|"
                           r"jackets?|dress|hair|smiling|holding|looking|carrying|talking|young|old)$")
_CARD_FILL = {"style": False, "place": True, "anatomy": False, "subject": True, "face": False}


def _doc_freq_terms(captions, vocab):
    """Her terimin kaç açıklamada geçtiği (belge sıklığı) — tek görselde 10 kez geçen kelime öne çıkmasın."""
    low = [" " + (c or "").lower() + " " for c in captions if c]
    out = []
    for w in vocab:
        pat = re.compile(r"(?<![a-z])" + re.escape(w) + r"s?(?![a-z])")
        df = sum(1 for c in low if pat.search(c))
        if df:
            out.append((df, w))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def _common_words(captions, min_share=0.25):
    """Açıklamaların en az %25'inde geçen sıradan olmayan kelimeler (konu/mekân için ortak özellikler)."""
    caps = [c for c in captions if c]
    if not caps:
        return []
    df = {}
    for c in caps:
        for w in set(re.findall(r"[a-z][a-z\-]{2,}", c.lower())):
            if w not in _CARD_STOP:
                df[w] = df.get(w, 0) + 1
    need = max(2, int(math.ceil(min_share * len(caps)))) if len(caps) >= 4 else 1
    return sorted(((n, w) for w, n in df.items() if n >= need), key=lambda x: (-x[0], x[1]))


def heuristic_card(captions, learn="style", n=14):
    """LLM yoksa: açıklamalarda en çok görselde geçen, öğretilen türe uygun kelimeleri seçer."""
    learn = norm_learn(learn)
    words, seen = [], set()
    vocab = _CARD_VOCAB[learn]
    if vocab:
        for _df, w in _doc_freq_terms(captions, vocab):
            key = w.rstrip("s")
            if key not in seen and len(words) < n:
                seen.add(key)
                words.append(w)
    if _CARD_FILL[learn] and len(words) < n:
        have = " " + " ".join(words) + " "
        for _df, w in _common_words(captions):
            if len(words) >= n:
                break
            if learn == "place" and (PERSON_WORDS.search(w) or _PEOPLE_TERMS.match(w)):
                continue                       # mekânda çevre öne çıksın: insanla ilgili kelimeler karta girmez
            key = w.rstrip("s")
            if key not in seen and (" " + key) not in have:
                seen.add(key)
                words.append(w)
    avoid = ANATOMY_AVOID if learn == "anatomy" else ""
    return {"style": ", ".join(words), "avoid": avoid, "summary": ""}


def heuristic_style_card(captions, n=14):
    return heuristic_card(captions, "style", n)


_CARD_SYSTEM = {
    "style": ("You analyze captions of images that share a common visual STYLE. Return ONLY a JSON object with: "
              "\"style\": 8-16 comma-separated English style tags describing what the images have in common (medium, "
              "technique, line work, color palette, lighting, composition, mood, level of detail); "
              "\"avoid\": 3-8 comma-separated things that would break this style; "
              "\"summary\": one short English sentence. Describe the style only — never name or describe specific "
              "people, characters, identities, brands or logos."),
    "place": ("You analyze captions of images that all show the same kind of PLACE (for example the streets of one "
              "city). Return ONLY a JSON object with: \"style\": 8-16 comma-separated English tags describing what "
              "this place looks like in common (architecture, street features, materials, landmarks, vegetation, "
              "typical vehicles and objects, light, weather, atmosphere). Focus on the environment, not on the people in "
              "it; \"avoid\": 3-8 comma-separated things that would not fit this place; \"summary\": one short "
              "English sentence. Never describe or identify specific people."),
    "subject": ("You analyze captions of images that all show the same SUBJECT (an object, product, animal or "
                "character). Return ONLY a JSON object with: \"style\": 8-16 comma-separated English tags describing "
                "the subject's consistent visual features (shape, colors, materials, parts, distinctive details); "
                "\"avoid\": 3-8 comma-separated things that would contradict them; \"summary\": one short English "
                "sentence. Never name real people, brands or logos."),
    "anatomy": ("You analyze captions of images collected to teach correct HUMAN ANATOMY and poses. Return ONLY a "
                "JSON object with: \"style\": 8-16 comma-separated English tags for the poses, viewpoints, framing "
                "and anatomical qualities the images have in common (for example full body, natural proportions, "
                "correct hand anatomy); \"avoid\": 3-8 comma-separated anatomy mistakes to avoid (for example extra "
                "fingers, fused fingers, extra limbs, bad proportions); \"summary\": one short English sentence. "
                "Never describe or identify specific people."),
    "face": ("You analyze captions of images that all show the same FACE (one person or character). Return ONLY a "
             "JSON object with: \"style\": 8-16 comma-separated English tags for the consistent facial and hair "
             "features (face shape, hair color and style, eye color, skin details, typical expression, glasses, "
             "facial hair); \"avoid\": 3-8 comma-separated things that would contradict them; \"summary\": one "
             "short English sentence. Never name or identify the person."),
}
_CARD_NAME_LABEL = {"style": "Style name", "place": "Place name", "subject": "Subject name", "anatomy": "Name",
                    "face": "Name"}


def card_request(captions, tag, learn="style"):
    learn = norm_learn(learn)
    caps = []
    total = 0
    for c in captions:
        c = (c or "").strip()[:220]
        if not c:
            continue
        if total + len(c) > 7000 or len(caps) >= 60:
            break
        caps.append("- " + c)
        total += len(c)
    user = f"{_CARD_NAME_LABEL[learn]} chosen by the user: {tag or '-'}\nCaptions:\n" + "\n".join(caps)
    schema = {"type": "object", "properties": {"style": {"type": "string"}, "avoid": {"type": "string"},
                                               "summary": {"type": "string"}}, "required": ["style", "avoid", "summary"]}
    return _CARD_SYSTEM[learn], user, schema


def style_card_request(captions, tag):
    return card_request(captions, tag, "style")


def save_card(paths, name, tag, card, dataset_slug, n_images, learn="style"):
    """Hafif modun sonucu: prompt'a eklenecek kelimeler ('style' alanı) + kaçınılacaklar, türüyle birlikte."""
    learn = norm_learn(learn)
    d = own_lora_dir(paths)
    os.makedirs(d, exist_ok=True)
    fn = safe_folder_name(name or tag or dataset_slug, learn) + ".card.json"
    body = {"name": name or tag or dataset_slug, "tag": tag, "style": merge_prompt_lists(card.get("style", ""), ""),
            "avoid": merge_prompt_lists(card.get("avoid", ""), ""), "summary": (card.get("summary") or "")[:300],
            "dataset": dataset_slug, "images": n_images, "created": time.strftime("%Y-%m-%d %H:%M"), "type": "card",
            "learn": learn}
    with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)
    return os.path.join(d, fn), body


def save_style_card(paths, name, tag, card, dataset_slug, n_images):
    return save_card(paths, name, tag, card, dataset_slug, n_images, "style")


# =============================================================================
# Ödül / ceza: kullanıcı puanları (👍/👎) ve basit otomatik hakem
# =============================================================================
PERSON_WORDS = re.compile(r"\b(woman|women|man|men|girl|boy|person|people|portrait|face|faces|hand|hands|child|"
                          r"kid|lady|guy|couple|family|selfie|character|warrior|knight|dancer|model)\b", re.I)


def feedback_path(paths):
    return os.path.join(paths.root, "feedback", "ratings.jsonl")


def add_feedback(paths, image_path, meta, rating):
    """rating: +1 (👍 ödül), -1 (👎 ceza), 0 (geri al). Son kayıt geçerlidir."""
    p = feedback_path(paths)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    meta = meta or {}
    rec = {"path": os.path.abspath(image_path), "rating": int(rating), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prompt": meta.get("prompt", ""), "negative": meta.get("negative", ""), "model_key": meta.get("model_key", ""),
           "seed": meta.get("seed"), "width": meta.get("width"), "height": meta.get("height")}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_feedback(paths):
    """{yol: son kayıt} — 0 puanlılar (geri alınanlar) dahil edilmez."""
    out = {}
    try:
        with open(feedback_path(paths), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["path"]] = r
                except Exception:
                    continue
    except OSError:
        return {}
    return {k: v for k, v in out.items() if v.get("rating") in (1, -1)}


def _norm_prompt(p):
    return re.sub(r"\s+", " ", (p or "").lower()).strip()


def feedback_training_data(paths, model_key="sd15", max_pairs_per_prompt=4):
    """Aynı prompt için 👍 ve 👎 alan görsellerden (kazanan, kaybeden) çiftleri; ayrıca tek kalan 👍'ler."""
    fb = [r for r in load_feedback(paths).values() if r.get("model_key") == model_key and os.path.exists(r["path"])]
    groups = {}
    for r in fb:
        groups.setdefault(_norm_prompt(r.get("prompt")), []).append(r)
    pairs, liked_single = [], []
    for _k, rs in groups.items():
        good = [r for r in rs if r["rating"] > 0]
        bad = [r for r in rs if r["rating"] < 0]
        made = 0
        for g in good:
            for b in bad:
                if made < max_pairs_per_prompt:
                    pairs.append((g, b))
                    made += 1
        if good and not bad:
            liked_single += good
    stats = {"liked": sum(1 for r in fb if r["rating"] > 0), "disliked": sum(1 for r in fb if r["rating"] < 0),
             "pairs": len(pairs)}
    return pairs, liked_single, stats


def judge_image(models_root, img, prompt=""):
    """
    Basit otomatik hakem (0–1). Yüz netliği (YuNet güveni), el algısı, keskinlik ve pozlamaya bakar.
    Kesin bir kalite ölçüsü DEĞİLDİR: bariz hataları (bozuk yüz, bulanık, kapkara) cezalandırmak içindir.
    """
    import numpy as np
    import cv2
    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    parts = {"sharp": min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 400.0)}
    dark, bright = float((gray < 12).mean()), float((gray > 243).mean())
    parts["exposure"] = max(0.0, 1.0 - (dark + bright) * 2.0)
    person = bool(PERSON_WORDS.search(prompt or ""))
    if person and detector_files_ok(models_root):
        try:
            det = get_detector(models_root)
            faces = det.faces(rgb, 0.3)
            parts["face"] = max((f[4] for f in faces), default=0.0)
            hands = det.hands(rgb, 0.5)
            if hands:
                parts["hand"] = sum(h[4] for h in hands) / len(hands)
        except Exception:
            pass
    weights = {"face": 0.5, "hand": 0.15, "sharp": 0.2, "exposure": 0.15}
    tw = sum(weights[k] for k in parts)
    score = sum(weights[k] * v for k, v in parts.items()) / tw if tw else 0.0
    return round(score, 4), parts


def collect_prompt_pool(paths, outdir, limit=300):
    """Kendi kendine öğrenme için prompt havuzu: puanlanan ve üretilen görsellerin prompt'ları."""
    seen, pool = set(), []
    for r in load_feedback(paths).values():
        k = _norm_prompt(r.get("prompt"))
        if k and k not in seen:
            seen.add(k)
            pool.append(r["prompt"])
    try:
        from PIL import Image
        files = sorted(glob.glob(os.path.join(outdir, "*.png")), key=os.path.getmtime, reverse=True)[:limit]
        for f in files:
            try:
                with Image.open(f) as im:
                    meta = json.loads(im.info.get("ails", "{}"))
                if meta.get("model_key") == "sd15" and meta.get("prompt"):
                    k = _norm_prompt(meta["prompt"])
                    if k not in seen:
                        seen.add(k)
                        pool.append(meta["prompt"])
            except Exception:
                continue
    except Exception:
        pass
    return pool[:limit]


DEFAULT_SELF_PROMPTS = [
    "portrait of a smiling woman, detailed face, natural light, highly detailed",
    "a man waving his hand, upper body, detailed face, detailed hands",
    "a girl holding a cup of coffee with both hands, cafe, soft light",
    "close-up of two hands playing piano, detailed fingers",
    "an old man reading a book, detailed face and hands, warm light",
    "a dancer raising her arms, full body, studio lighting",
]


# =============================================================================
# LoRA eğitimi (SD 1.5) — stil öğrenme (normal eğitim) ve ödül/ceza (Diffusion-DPO)
#   4 GB VRAM'e sığacak şekilde: görüntü kodları ve metin gömmeleri önceden hesaplanır, VAE ve metin
#   kodlayıcı RAM'e alınır; UNet fp16 + gradyan kontrol noktaları; LoRA ağırlıkları fp32.
#   DPO bellek dostu uygulanır: önce gradyansız ileri geçişlerle katsayı bulunur, sonra kazanan ve
#   kaybeden için AYRI geri yayılım yapılır (tek seferde ikisini tutmak gerekmez).
# =============================================================================
TRAIN_BASES = ("sd15", "sdxl", "flux2_klein_4b", "zimage_turbo")
# Eğitim modelleri. vram: güvenli çalışması için gereken en az VRAM (GB). FLUX.2 ve Z-Image, LoRA'yı damıtılmış
# (hızlı) modelde değil, aynı ailenin damıtılmamış "base" modelinde eğitir; eğitilen LoRA hızlı modelde kullanılır.
# Base modelden yalnızca dönüştürücü (transformer) indirilir; VAE, metin kodlayıcı ve sözlük kurulu modelden alınır.
TRAIN_SPECS = {
    "sd15": {"vram": 3.5, "res": 512, "fallback": (384,), "cpu_ok": True, "ram": 8},
    "sdxl": {"vram": 10.0, "res": 1024, "fallback": (768, 512), "cpu_ok": False, "ram": 16, "res_swap": 768},
    "flux2_klein_4b": {"vram": 12.0, "res": 512, "fallback": (384,), "cpu_ok": False, "ram": 24,
                       "base_repo": "black-forest-labs/FLUX.2-klein-base-4B", "base_name": "FLUX.2 [klein] base 4B",
                       "base_gb": 7.8, "license": "Apache-2.0", "sample_steps": 24, "sample_cfg": 4.0},
    "zimage_turbo": {"vram": 16.0, "res": 512, "fallback": (384,), "cpu_ok": False, "ram": 28,
                     "base_repo": "Tongyi-MAI/Z-Image", "base_name": "Z-Image (base)", "base_gb": 12.3,
                     "license": "Apache-2.0", "sample_steps": 28, "sample_cfg": 4.0},
}


SWAP_MIN_VRAM = 3.5           # RAM destekli eğitim için en az VRAM (4 GB kartlar)


def train_ready(hw, key):
    """
    (uygun_mu, durum) — durum: '' (tamamen ekran kartında) | 'swap' (RAM destekli, yavaş) |
    'vram' (NVIDIA GPU yok ya da 3.5 GB'tan az) | 'ram' (RAM yetmez).
    """
    sp = TRAIN_SPECS[key]
    if sp["cpu_ok"]:
        return True, ""
    if not hw.cuda_usable or hw.vram_gb < SWAP_MIN_VRAM - 0.3:
        return False, "vram"
    if hw.vram_gb >= sp["vram"] - 0.3:
        return True, ""
    if hw.ram_total_gb + 0.5 < sp["ram"]:
        return False, "ram"
    return True, "swap"


def train_base_dir(models_root, key):
    """Eğitimde kullanılan dönüştürücünün klasörü: base modeli varsa onun, yoksa kurulu modelin kendisi."""
    sp = TRAIN_SPECS[key]
    if sp.get("base_repo"):
        return os.path.join(models_root, "train", key)
    return image_model_dir(models_root, IMAGE_MODEL_BY_KEY[key])


def is_train_base_installed(models_root, key):
    if not TRAIN_SPECS[key].get("base_repo"):
        return is_image_model_installed(models_root, IMAGE_MODEL_BY_KEY[key])
    return os.path.exists(os.path.join(train_base_dir(models_root, key), MARKER))


def download_train_base(models_root, key, progress_cb=None, abort_event=None):
    """Base modelin yalnızca dönüştürücüsü (transformer/*) indirilir."""
    from huggingface_hub import snapshot_download
    sp = TRAIN_SPECS[key]
    dest = train_base_dir(models_root, key)
    allow = ["transformer/*.json", "transformer/*.safetensors"]
    total = int(sp["base_gb"] * GB)
    try:
        infos = snapshot_download(sp["base_repo"], allow_patterns=allow, dry_run=True)
        total = sum((i.file_size or 0) for i in infos) or total
    except Exception:
        pass
    already = folder_size_bytes(dest) if os.path.isdir(dest) else 0
    check_space(models_root, max(0, total - already) / GB + 1.0)
    os.makedirs(dest, exist_ok=True)
    result = {"err": None}

    def _run():
        try:
            snapshot_download(sp["base_repo"], local_dir=dest, allow_patterns=allow, max_workers=4)
        except Exception as e:  # noqa
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(1.0)
        if progress_cb:
            progress_cb(folder_size_bytes(dest), total)
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
    if result["err"] is not None:
        err = result["err"]
        if isinstance(err, OSError) and getattr(err, "errno", None) == 28:
            raise DiskFullError(drive_label(models_root), total / GB, disk_free_gb(models_root))
        raise err
    if not os.path.exists(os.path.join(dest, "transformer", "config.json")):
        raise RuntimeError("transformer/config.json missing — download incomplete")
    with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
        json.dump({"repo": sp["base_repo"], "size_gb": round(folder_size_bytes(dest) / GB, 2),
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f)
    return dest


def make_trainer(paths, key, device, rank, lr, resolution=None):
    model = IMAGE_MODEL_BY_KEY[key]
    path = image_model_dir(paths.models, model)
    if key == "sd15":
        return LoraTrainer(path, model, device, rank=rank, lr=lr, resolution=resolution)
    if key == "sdxl":
        return SDXLLoraTrainer(path, model, device, rank=rank, lr=lr, resolution=resolution)
    return FlowLoraTrainer(path, train_base_dir(paths.models, key), model, device, rank=rank, lr=lr,
                           resolution=resolution)
STYLE_RANK, PREF_RANK = 16, 8
STYLE_LR, PREF_LR = 1e-4, 1e-5

# Ne öğretiliyor? Her tür kendi ayarlarıyla eğitilir:
#   views: görselin hangi kesitleri öğretilir — "c" orta kare, "a"/"b" geniş/uzun görselin iki ucu, "f" aynalı orta
#   (şehir/mekân görüntülerinde yazılar ve trafik yönü ters dönmesin diye ayna yok; geniş video karelerinden
#   iki uç kesit de alınır. Konu/nesnede logo ve asimetrik detaylar bozulmasın diye ayna yok.)
LEARN_KINDS = ("style", "place", "subject", "anatomy", "face")
LEARN_PRESETS = {
    "style":   {"icon": "🎨", "rank": 16, "lr": 1e-4, "noise_offset": 0.05, "views": ("c", "f")},
    "place":   {"icon": "🏙️", "rank": 16, "lr": 1e-4, "noise_offset": 0.03, "views": ("c", "a", "b")},
    "subject": {"icon": "🧸", "rank": 16, "lr": 1e-4, "noise_offset": 0.0, "views": ("c",)},
    "anatomy": {"icon": "🧍", "rank": 16, "lr": 5e-5, "noise_offset": 0.0, "views": ("c", "f")},
    # yüz: her görselden yalnızca en büyük yüz (saçla birlikte) kesilir; ayna yok (ben, saç ayrımı gibi
    # asimetrik ayrıntılar kimliğin parçası)
    "face":    {"icon": "👤", "rank": 16, "lr": 1e-4, "noise_offset": 0.0, "views": ("face",)},
}
FACE_PROMPT = "close-up face portrait"
FACE_MIN_PX = 40              # bundan küçük yüzler öğretilmez (çok bulanık olur)
FACE_BG_WEIGHT = 0.2          # yüz kesitinde yüz dışı (arka plan, boyun, kıyafet) bu ağırlıkla öğretilir


PLACE_PERSON_WEIGHT = 0.35    # mekân eğitiminde insan gövdeleri bu ağırlıkla öğretilir (çevre öne çıksın)
HIDE_FACE_SCORE = 0.45        # gizlerken eşik biraz düşük: kaçan yüz olmasın (yanlış alarm yalnızca biraz çevre kaybettirir)


def find_people(models_root, img):
    """Yüz kutuları (YuNet + anime): [(x, y, g, y), ...] — görselin kendi piksel ölçeğinde."""
    import numpy as np
    det = get_detector(models_root)
    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    return [(float(d[0]), float(d[1]), float(d[2]), float(d[3])) for d in det.faces(rgb, HIDE_FACE_SCORE)]


def hide_people(img, faces):
    """
    Mekân eğitimi için: yüzleri tanınmayacak biçimde bulanıklaştırır ve bir ağırlık haritası döndürür.
    Yüz (saç/çene payıyla) = 0 → hiç öğretilmez (bulanık yüz de öğretilmez); yüzden tahmin edilen gövde = 0.35 →
    insanlar az öğretilir; geri kalan çevre = 1 → asıl öğretilen.
    """
    import numpy as np
    from PIL import ImageFilter
    W, H = img.size
    wmap = np.ones((H, W), np.float32)
    out = img.copy()
    for (x, y, w, h) in faces:
        cx, cy = x + w / 2.0, y + h / 2.0
        bx0, bx1 = int(max(0, cx - 1.8 * w)), int(min(W, cx + 1.8 * w))      # omuzlar ~3.6 yüz genişliği
        by0, by1 = int(max(0, y - 0.4 * h)), int(min(H, y + 8.0 * h))         # boy ~8 yüz yüksekliği
        if bx1 > bx0 and by1 > by0:
            wmap[by0:by1, bx0:bx1] = np.minimum(wmap[by0:by1, bx0:bx1], PLACE_PERSON_WEIGHT)
        fx0, fx1 = int(max(0, cx - 0.8 * w)), int(min(W, cx + 0.8 * w))
        fy0, fy1 = int(max(0, cy - 0.95 * h)), int(min(H, cy + 0.8 * h))
        if fx1 > fx0 and fy1 > fy0:
            wmap[fy0:fy1, fx0:fx1] = 0.0
            region = out.crop((fx0, fy0, fx1, fy1))
            small = region.resize((max(1, region.width // 10), max(1, region.height // 10)))
            region = small.resize(region.size).filter(ImageFilter.GaussianBlur(max(3, max(region.size) // 6)))
            out.paste(region, (fx0, fy0))
    return out, wmap


def face_crop(img, box):
    """Yüz modu: yüzü saç ve çeneyle birlikte kare olarak keser. Dönüş: (kesit, ağırlık haritası) — yüz+saç 1,
    geri kalan (arka plan, boyun, kıyafet) FACE_BG_WEIGHT; yani yalnızca yüz okunur."""
    import numpy as np
    W, H = img.size
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0 - 0.1 * h
    side = max(1, min(int(round(max(w, h) * 2.4)), W, H))     # yüz + saç + biraz omuz; görsele sığacak kadar
    left = int(round(min(max(0, cx - side / 2.0), W - side)))
    top = int(round(min(max(0, cy - side / 2.0), H - side)))
    crop = img.crop((left, top, left + side, top + side))
    wmap = np.full((side, side), FACE_BG_WEIGHT, np.float32)
    fx0, fx1 = int(max(0, cx - 0.8 * w - left)), int(min(side, cx + 0.8 * w - left))
    fy0, fy1 = int(max(0, (y + h / 2.0) - 0.95 * h - top)), int(min(side, (y + h / 2.0) + 0.8 * h - top))
    if fx1 > fx0 and fy1 > fy0:
        wmap[fy0:fy1, fx0:fx1] = 1.0
    return crop, wmap


def _lora_sidecar(path):
    try:
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


TRAIN_SUFFIX = {"sd15": "sd15", "sdxl": "sdxl", "flux2_klein_4b": "flux2", "zimage_turbo": "zimage"}


class TrainVramError(RuntimeError):
    """Seçilen eğitim modeli için ekran kartı belleği yetersiz (ya da NVIDIA GPU yok)."""


def pick_lora_path(d, name, slug, learn, rank, base="sd15"):
    """
    Eğitimin kaydedileceği dosya. Aynı veri seti + tür + ayarla eğitilmiş bir LoRA varsa ondan devam edilir;
    yoksa ilk BOŞ ad seçilir (ad, ad_place, ad_place_2 …). Başka bir eğitime ait LoRA asla ezilmez.
    Dönüş: (yol, önceki_bilgi — devam edilecekse dolu, değilse {}).
    """
    cands = [name, f"{name}_{learn}"] + [f"{name}_{learn}_{i}" for i in range(2, 100)]
    free = None
    for c in cands:
        p = os.path.join(d, c + ".safetensors")
        if os.path.exists(p):
            info = _lora_sidecar(p)
            if info.get("dataset") == slug and info.get("rank") == rank and \
                    norm_learn(info.get("learn") or info.get("type")) == learn and \
                    (info.get("base") or "sd15") == base:
                return p, info
        elif free is None and not os.path.exists(os.path.splitext(p)[0] + ".json"):
            free = p
    return free or os.path.join(d, f"{name}_{learn}_{int(time.time())}.safetensors"), {}


def norm_learn(kind):
    return kind if kind in LEARN_KINDS else "style"


def learn_icon(kind):
    return LEARN_PRESETS[norm_learn(kind)]["icon"]


def image_views(w, h, wanted):
    """Bir görselden öğretilecek kesitler. Uç kesitler yalnızca belirgin geniş/uzun görsellerde (oran ≥ 1.3) alınır."""
    out = []
    long_ratio = max(w, h) / float(max(1, min(w, h)))
    for v in wanted:
        if v in ("a", "b") and long_ratio < 1.3:
            continue
        out.append(v)
    return out or ["c"]
DPO_BETA = 2500.0
PREF_LORA_NAME = "ails_tercih_sd15"


def ascii_name(text, maxlen=40):
    """'İstanbul Sokakları' → 'istanbul_sokaklari' (Türkçe harfler sadeleştirilir; Korece vb. için kısa özet)."""
    import hashlib
    import unicodedata
    tr = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")
    s = unicodedata.normalize("NFKD", (text or "").translate(tr)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()[:maxlen].strip("_")
    if not s:
        s = "ails" + hashlib.sha1((text or "x").encode("utf-8")).hexdigest()[:6]
    return s


def ascii_trigger(tag, learn="style"):
    """Metin kodlayıcının iyi anladığı ASCII tetikleyici: stil için 'manga_tarzi style', diğerleri için
    'istanbul_sokak' (bilinen kelimeler — 'istanbul' gibi — modelin önceki bilgisini de kullanır)."""
    s = ascii_name(tag)
    return s + " style" if learn == "style" else s


def train_device(hw):
    return hw.device if (hw.cuda_usable and hw.vram_gb >= 3.5) else "cpu"


class LoraTrainer:
    def __init__(self, model_path, model, device, rank=STYLE_RANK, lr=STYLE_LR, resolution=None):
        torch = import_torch()
        import contextlib
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig
        self.torch, self._ctx = torch, contextlib
        self.device = torch.device(device)
        self.on_gpu = self.device.type == "cuda"
        self.wdtype = torch.float16 if self.on_gpu else torch.float32
        self.res = int(resolution or model["base_px"])
        kw = dict(torch_dtype=self.wdtype)
        variant = model.get("variant")

        def _load(cls, sub):
            try:
                return cls.from_pretrained(model_path, subfolder=sub, variant=variant, **kw) if variant \
                    else cls.from_pretrained(model_path, subfolder=sub, **kw)
            except (OSError, ValueError):
                return cls.from_pretrained(model_path, subfolder=sub, **kw)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = _load(CLIPTextModel, "text_encoder")
        self.vae = _load(AutoencoderKL, "vae")
        self.unet = _load(UNet2DConditionModel, "unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        for m in (self.text_encoder, self.vae, self.unet):
            m.requires_grad_(False)
        self.lora_config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                      target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        self.unet.add_adapter(self.lora_config)
        for p in self.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()          # LoRA ağırlıkları fp32 (kararlı eğitim)
        self.params = [p for p in self.unet.parameters() if p.requires_grad]
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception:
            pass
        self.unet.to(self.device)
        self.unet.train()
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-2)
        self.scaler = torch.amp.GradScaler("cuda") if self.on_gpu else None
        self.pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")
        self.steps = 0
        self._emb_cache = {}
        self.noise_offset = 0.05      # stil için koyu/aydınlık tonları da öğrenebilsin; tür ayarı değiştirir
        try:                          # görsel → gizli uzay küçültme oranı (SD 1.5'te 8)
            self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        except Exception:
            self.vae_scale = 8

    # ---- yardımcılar
    def _autocast(self):
        if self.on_gpu:
            return self.torch.autocast("cuda", dtype=self.torch.float16)
        return self._ctx.nullcontext()

    def _geometry(self, w, h, view):
        """Kısa kenarı eğitim boyutuna getiren ölçek ve kare kesitin yeri: (yeni_g, yeni_y, sol, üst)."""
        s = self.res / float(min(w, h))
        nw, nh = max(self.res, int(round(w * s))), max(self.res, int(round(h * s)))
        if view == "a":
            left, top = 0, 0
        elif view == "b":
            left, top = nw - self.res, nh - self.res
        else:
            left, top = (nw - self.res) // 2, (nh - self.res) // 2
        return nw, nh, left, top

    def _prep(self, img, view="c"):
        """Kısa kenar eğitim boyutuna ölçeklenir, kare kesit alınır. view: c=orta, a=sol/üst uç, b=sağ/alt uç,
        f=aynalanmış orta."""
        import numpy as np
        from PIL import Image
        torch = self.torch
        img = img.convert("RGB")
        nw, nh, left, top = self._geometry(img.width, img.height, view)
        img = img.resize((nw, nh), Image.BICUBIC).crop((left, top, left + self.res, top + self.res))
        if view == "f":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img).astype("float32") / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _people_tensors(self, img, view, people):
        """Yüz modu ('face'): yalnızca yüz kesiti; mekân: yüzler gizli. Dönüş: (görsel, ağırlık)."""
        if view == "face":
            crop, wmap = face_crop(img, people[0])
            return self._prep(crop, "c"), self._prep_weight(wmap, "c")
        shown, wmap = hide_people(img, people)
        return self._prep(shown, view), self._prep_weight(wmap, view)

    def _prep_weight(self, wmap, view="c"):
        """Ağırlık haritasını görselle AYNI ölçek/kesit/aynayla dönüştürüp gizli uzay boyutuna (SD 1.5'te 1/8) indirir.
        Küçültmede en düşük değer alınır: yüzün bir pikseli bile düşen hücre tamamen öğretimden çıkar."""
        import numpy as np
        from PIL import Image
        torch = self.torch
        h, w = wmap.shape
        nw, nh, left, top = self._geometry(w, h, view)
        m = Image.fromarray((np.clip(wmap, 0, 1) * 255).astype("uint8"), "L").resize((nw, nh), Image.NEAREST)
        m = np.asarray(m.crop((left, top, left + self.res, top + self.res))).astype("float32") / 255.0
        if view == "f":
            m = m[:, ::-1]
        k = self.vae_scale
        n = self.res // k
        m = m[:n * k, :n * k].reshape(n, k, n, k).min(axis=(1, 3))
        return torch.from_numpy(np.ascontiguousarray(m)).view(1, 1, n, n).to(torch.float16)

    def encode_images(self, imgs, progress_cb=None, abort_event=None, skip_bad=False, with_weights=False):
        """imgs: PIL görselleri, dosya yolları, (yol|PIL, kesit) ya da (yol|PIL, kesit, insanlar) üçlüleri.
        'insanlar' verilirse yüzler bulanıklaştırılır ve ağırlık haritası üretilir (bkz. hide_people).
        Yollar tek tek açılır (bellekte hepsi tutulmaz). skip_bad=True ise açılamayanlar için None döner.
        with_weights=True ise (gizli_kodlar, ağırlıklar) döner; ağırlığı olmayanlar için None."""
        from PIL import Image
        torch = self.torch
        out, weights = [], []
        self.vae.to(self.device)
        try:
            with torch.no_grad():
                for n, im in enumerate(imgs):
                    if abort_event is not None and abort_event.is_set():
                        raise AbortedError("user")
                    view, people = "c", None
                    if isinstance(im, tuple):
                        if len(im) == 3:
                            im, view, people = im
                        else:
                            im, view = im
                    wt = None
                    try:
                        if isinstance(im, str):
                            with Image.open(im) as f:
                                if people is None:
                                    f.draft("RGB", (self.res * 2, self.res * 2))
                                    x = self._prep(f, view)
                                else:
                                    x, wt = self._people_tensors(f.convert("RGB"), view, people)
                        elif people is not None:
                            x, wt = self._people_tensors(im.convert("RGB"), view, people)
                        else:
                            x = self._prep(im, view)
                    except Exception:
                        if not skip_bad:
                            raise
                        out.append(None)
                        weights.append(None)
                        continue
                    lat = self._encode_latent(x)
                    out.append(lat.to("cpu"))
                    weights.append(wt)
                    if progress_cb:
                        progress_cb(n + 1, len(imgs))
        finally:
            self.vae.to("cpu")
            free_gpu_memory()
        return (out, weights) if with_weights else out

    def _abort_cb(self, _pipe, _i, _t, kw):
        """Örnek görsel üretirken Durdur'a basılırsa hemen çık."""
        ab = getattr(self, "abort_event", None)
        if ab is not None and ab.is_set():
            raise AbortedError("user")
        return kw

    def _encode_latent(self, x):
        """Görsel → modelin gizli uzayı (SD 1.5: VAE örneği × ölçek)."""
        return self.vae.encode(x.to(self.device, self.wdtype)).latent_dist.sample() * self.vae.config.scaling_factor

    def _encode_text_on_device(self, text):
        torch = self.torch
        if text in self._emb_cache:
            return self._emb_cache[text]
        with torch.no_grad():
            ids = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
            emb = self.text_encoder(ids)[0].to("cpu")
        if len(self._emb_cache) < 4000:
            self._emb_cache[text] = emb
        return emb

    def encode_text(self, text):
        if text in self._emb_cache:
            return self._emb_cache[text]
        self.text_encoder.to(self.device)
        try:
            return self._encode_text_on_device(text)
        finally:
            self.text_encoder.to("cpu")

    def encode_texts(self, texts, progress_cb=None):
        self.text_encoder.to(self.device)
        try:
            out = []
            for n, t in enumerate(texts):
                out.append(self._encode_text_on_device(t))
                if progress_cb:
                    progress_cb(n + 1, len(texts))
            return out
        finally:
            self.text_encoder.to("cpu")
            free_gpu_memory()

    def _target(self, lat, noise, t):
        if self.pred_type == "v_prediction":
            return self.noise_scheduler.get_velocity(lat, noise, t)
        return noise

    def _mse(self, pred, target):
        return ((pred.float() - target.float()) ** 2).mean()

    def _backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _opt_step(self):
        torch = self.torch
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.steps += 1

    # ---- normal eğitim adımı (stil öğrenme, 👍'lenen tek görseller)
    def step_sft(self, lat, emb, noise=None, t=None, weight=None):
        """Normal eğitim adımı. weight: (1,1,h,w) ağırlık haritası — 0 olan yerler (ör. gizlenen yüzler) hiç
        öğretilmez, düşük olanlar (ör. insanlar) daha az öğretilir; böylece mekânda çevre öne çıkar."""
        torch = self.torch
        lat = lat.to(self.device, self.wdtype)
        emb = emb.to(self.device, self.wdtype)
        if noise is None:
            noise = torch.randn_like(lat)
            if self.noise_offset:
                noise = noise + self.noise_offset * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=lat.device,
                                                                dtype=lat.dtype)
        if t is None:
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=lat.device)
        noisy = self.noise_scheduler.add_noise(lat, noise, t.long())
        with self._autocast():
            pred = self.unet(noisy, t, emb).sample
        target = self._target(lat, noise, t)
        if weight is None:
            loss = self._mse(pred, target)
        else:
            w = weight.to(self.device, torch.float32)
            err = (pred.float() - target.float()) ** 2
            loss = (err * w).sum() / (w.sum() * err.shape[1]).clamp_min(1e-6)
        self._backward(loss)
        self._opt_step()
        return float(loss.detach())

    # ---- ödül/ceza adımı (Diffusion-DPO), bellek dostu
    def dpo_terms(self, lat_w, lat_l, emb, noise, t):
        torch = self.torch
        noisy_w = self.noise_scheduler.add_noise(lat_w, noise, t)
        noisy_l = self.noise_scheduler.add_noise(lat_l, noise, t)
        tw, tl = self._target(lat_w, noise, t), self._target(lat_l, noise, t)
        with torch.no_grad(), self._autocast():
            m_w = self._mse(self.unet(noisy_w, t, emb).sample, tw)
            m_l = self._mse(self.unet(noisy_l, t, emb).sample, tl)
            self.unet.disable_adapters()
            try:
                r_w = self._mse(self.unet(noisy_w, t, emb).sample, tw)
                r_l = self._mse(self.unet(noisy_l, t, emb).sample, tl)
            finally:
                self.unet.enable_adapters()
        return noisy_w, noisy_l, tw, tl, m_w, m_l, r_w, r_l

    def step_dpo(self, lat_w, lat_l, emb, beta=DPO_BETA, noise=None, t=None, apply=True):
        """Dönüş: (kayıp, ödül doğruluğu 0/1). Kazanan ve kaybeden aynı gürültü ve adımla değerlendirilir."""
        torch = self.torch
        import torch.nn.functional as F
        lat_w = lat_w.to(self.device, self.wdtype)
        lat_l = lat_l.to(self.device, self.wdtype)
        emb = emb.to(self.device, self.wdtype)
        if noise is None:
            noise = torch.randn_like(lat_w)
        if t is None:
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=lat_w.device)
        t = t.long()
        noisy_w, noisy_l, tw, tl, m_w, m_l, r_w, r_l = self.dpo_terms(lat_w, lat_l, emb, noise, t)
        z = -0.5 * beta * ((m_w - m_l) - (r_w - r_l))
        loss = float(-F.logsigmoid(z))
        coef = float(0.5 * beta * torch.sigmoid(-z))     # dL/dθ = coef·(∂m_w/∂θ − ∂m_l/∂θ)
        if apply:
            with self._autocast():
                lw = self._mse(self.unet(noisy_w, t, emb).sample, tw)
            self._backward(coef * lw)
            with self._autocast():
                ll = self._mse(self.unet(noisy_l, t, emb).sample, tl)
            self._backward(-coef * ll)
            self._opt_step()
        return loss, 1.0 if float(z) > 0 else 0.0

    # ---- kaydet / devam et / örnek görsel
    def lora_state(self):
        from peft.utils import get_peft_model_state_dict
        from diffusers.utils import convert_state_dict_to_diffusers
        return convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))

    def save(self, out_path, info):
        from diffusers import StableDiffusionPipeline
        d, fn = os.path.split(out_path)
        os.makedirs(d, exist_ok=True)
        check_space(d, 0.2)
        part = fn + ".part"
        StableDiffusionPipeline.save_lora_weights(save_directory=d, unet_lora_layers=self.lora_state(),
                                                  weight_name=part, safe_serialization=True)
        os.replace(os.path.join(d, part), out_path)
        with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        return out_path

    def load_state(self, path):
        """Önceki eğitimden devam (ör. tercih LoRA'sı her seferinde birikerek gelişir)."""
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        from diffusers.utils import convert_unet_state_dict_to_peft
        sd = load_file(path)
        sd = {k[len("unet."):]: v for k, v in sd.items() if k.startswith("unet.")}
        res = set_peft_model_state_dict(self.unet, convert_unet_state_dict_to_peft(sd), adapter_name="default")
        for p in self.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return res

    def sample(self, prompt, negative="", seed=1234, steps=20, cfg=7.0, size=None):
        torch = self.torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        sched = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config, use_karras_sigmas=True,
                                                        algorithm_type="dpmsolver++")
        pipe = StableDiffusionPipeline(vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                                       unet=self.unet, scheduler=sched, safety_checker=None, feature_extractor=None,
                                       requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        size = int(size or self.res)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.eval()
        try:
            with torch.no_grad(), self._autocast():
                img = pipe(prompt, negative_prompt=negative or None, num_inference_steps=steps, guidance_scale=cfg,
                           width=size, height=size, generator=torch.Generator("cpu").manual_seed(int(seed)),
                           callback_on_step_end=self._abort_cb).images[0]
        finally:
            self.unet.train()
            self.vae.to("cpu")
            self.text_encoder.to("cpu")
            free_gpu_memory()
        return img

    def close(self):
        self.unet = self.vae = self.text_encoder = None
        self.opt = None
        free_gpu_memory()


class ParamSwapper:
    """
    RAM destekli çalıştırma ("blok takası"): modelin donmuş ağırlıkları RAM'de durur; bir parça (blok) çalışacağı
    an ekran kartına kopyalanır, iş bitince RAM'deki kopyasına geri bağlanır. Eğitilen LoRA ağırlıkları, tamponlar ve
    küçük parçalar hep ekran kartında kalır. Ekran kartında yer kaldıkça baştan itibaren parçalar kalıcı tutulur
    (kart büyüdükçe hızlanır). Geri yayılım (backward), ileri geçişte kopyalanan GPU tensörlerini kullanır; gradyan
    kontrol noktaları (checkpointing) her bloğu geri yayılımda yeniden çalıştırdığında kanca yeniden kopyalar.
    """
    SMALL = 32 * MB              # bundan küçük parçalar her zaman ekran kartında
    MAX_UNIT = 400 * MB          # bundan büyük parçalar alt parçalarına bölünür

    def __init__(self, model, device, budget_bytes, pin=True, small=None, max_unit=None, abort_event=None):
        torch = import_torch()
        self.abort_event = abort_event
        if small is not None:
            self.SMALL = small
        if max_unit is not None:
            self.MAX_UNIT = max_unit
        self.torch = torch
        self.device = torch.device(device)
        self.model = model
        self.handles = []
        self.units = []                       # [(modül, [(parametre, ram_kopyası)])]
        seen = set()
        units = []

        def frozen_bytes(m, recurse=True):
            return sum(p.numel() * p.element_size() for p in m.parameters(recurse=recurse)
                       if not p.requires_grad and id(p) not in seen)

        def walk(m):
            own = [p for p in m.parameters(recurse=False) if not p.requires_grad and id(p) not in seen]
            total = frozen_bytes(m)
            if total <= self.MAX_UNIT or not list(m.children()):
                ps = [p for p in m.parameters() if not p.requires_grad and id(p) not in seen]
                for p in ps:
                    seen.add(id(p))
                if ps:
                    units.append((m, ps))
                return
            if own:                            # alt modüllere inen bir modülün kendi doğrudan ağırlıkları
                for p in own:
                    seen.add(id(p))
                units.append((m, own))
            for c in m.children():
                walk(c)

        walk(model)
        if pin:
            # Sabitlenmiş (pinned) bellek kopyayı hızlandırır ama Windows onu diske yazamaz; büyükse Windows diğer
            # her şeyi sayfa dosyasına iter. Bu yüzden yalnızca RAM'in küçük bir payıysa kullanılır.
            total = sum(sum(p.numel() * p.element_size() for p in ps) for _m, ps in units)
            pin = (total - budget_bytes) < ram_total_gb() * GB * 0.20
        for b in model.buffers():
            b.data = b.data.to(self.device)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(self.device)
        used = 0
        swapped_bytes = 0
        self.resident_bytes = 0
        for m, ps in units:
            size = sum(p.numel() * p.element_size() for p in ps)
            if size <= self.SMALL or used + size <= budget_bytes:
                for p in ps:
                    p.data = p.data.to(self.device)
                used += size
                continue
            pairs = []
            for p in ps:
                cpu = p.data.to("cpu")
                if pin:
                    try:
                        cpu = cpu.pin_memory()
                    except Exception:
                        pin = False
                p.data = cpu
                pairs.append((p, cpu))
            swapped_bytes += size
            self.units.append((m, pairs))
            self.handles.append(m.register_forward_pre_hook(self._make_pre(pairs)))
            self.handles.append(m.register_forward_hook(self._make_post(pairs)))
        self.resident_bytes = used
        self.swapped_bytes = swapped_bytes

    def _make_pre(self, pairs):
        dev = self.device

        def pre(_m, _args):
            ab = self.abort_event
            if ab is not None and ab.is_set():
                raise AbortedError("user")      # Durdur'a basıldı: adımın ortasında bile hemen çık
            for p, cpu in pairs:
                p.data = cpu.to(dev, non_blocking=True)
        return pre

    def _make_post(self, pairs):
        def post(_m, _args, _out):
            for p, cpu in pairs:
                p.data = cpu          # GPU kopyası, geri yayılım için gerekiyorsa grafikte yaşamaya devam eder
        return post

    def remove(self, to="cpu"):
        for h in self.handles:
            h.remove()
        self.handles = []
        for _m, pairs in self.units:
            for p, cpu in pairs:
                p.data = cpu
        self.units = []
        if to is not None:
            self.model.to(to)
        gc.collect()
        release_pinned_cache()


def release_pinned_cache():
    """PyTorch'un önbellekte tuttuğu sabitlenmiş RAM bloklarını Windows'a geri verir."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    for fn in ("_host_emptyCache", "_accelerator_emptyHostCache"):
        try:
            getattr(torch._C, fn)()
            return
        except Exception:
            continue


def gpu_free_bytes(device):
    torch = import_torch()
    try:
        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return int(free)
    except Exception:
        return 0


def module_bytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters())


class SDXLLoraTrainer(LoraTrainer):
    """SDXL LoRA eğitimi: iki metin kodlayıcı, ek zaman/boyut koşulları. VAE fp32 kodlar (SDXL VAE fp16'da bozulur)."""

    def __init__(self, model_path, model, device, rank=STYLE_RANK, lr=STYLE_LR, resolution=None):
        torch = import_torch()
        import contextlib
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
        from peft import LoraConfig
        self.torch, self._ctx = torch, contextlib
        self.device = torch.device(device)
        self.on_gpu = self.device.type == "cuda"
        self.wdtype = torch.float16 if self.on_gpu else torch.float32
        self.res = int(resolution or TRAIN_SPECS["sdxl"]["res"])
        kw = dict(torch_dtype=self.wdtype)
        variant = model.get("variant")

        def _load(cls, sub, **extra):
            k = dict(kw, **extra)
            try:
                return cls.from_pretrained(model_path, subfolder=sub, variant=variant, **k) if variant \
                    else cls.from_pretrained(model_path, subfolder=sub, **k)
            except (OSError, ValueError):
                return cls.from_pretrained(model_path, subfolder=sub, **k)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.tokenizer_2 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        self.text_encoder = _load(CLIPTextModel, "text_encoder")
        self.text_encoder_2 = _load(CLIPTextModelWithProjection, "text_encoder_2")
        self.vae = _load(AutoencoderKL, "vae", torch_dtype=torch.float32)
        self.unet = _load(UNet2DConditionModel, "unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        for m in (self.text_encoder, self.text_encoder_2, self.vae, self.unet):
            m.requires_grad_(False)
        self.lora_config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                      target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        self.unet.add_adapter(self.lora_config)
        for p in self.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.params = [p for p in self.unet.parameters() if p.requires_grad]
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception:
            pass
        self.unet.train()
        self._on_device = False
        self.swapper = None
        self.swap_info = None
        self.reserve = int(2.5 * GB)
        if self.on_gpu and resolution is None and \
                module_bytes(self.unet) + self.reserve > gpu_free_bytes(self.device):
            self.res = int(TRAIN_SPECS["sdxl"]["res_swap"])      # RAM destekli: daha küçük ara sonuçlar
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-2)
        self.scaler = torch.amp.GradScaler("cuda") if self.on_gpu else None
        self.pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")
        self.steps = 0
        self._emb_cache = {}
        self.noise_offset = 0.05
        try:
            self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        except Exception:
            self.vae_scale = 8

    def _ensure_unet(self):
        if self._on_device:
            return
        self._on_device = True
        if not self.on_gpu:
            self.unet.to(self.device)
            return
        free_gpu_memory()
        free = gpu_free_bytes(self.device)
        if module_bytes(self.unet) + self.reserve <= free:
            self.unet.to(self.device)
            return
        self.swapper = ParamSwapper(self.unet, self.device, max(0, free - self.reserve),
                                    abort_event=getattr(self, "abort_event", None))
        self.swap_info = (self.swapper.resident_bytes / GB, self.swapper.swapped_bytes / GB)

    def prepare_device(self):
        self._ensure_unet()
        return self.swap_info

    def _encode_latent(self, x):
        return self.vae.encode(x.to(self.device, self.torch.float32)).latent_dist.sample() * self.vae.config.scaling_factor

    def _encode_text_on_device(self, text):
        torch = self.torch
        if text in self._emb_cache:
            return self._emb_cache[text]
        with torch.no_grad():
            embs = []
            pooled = None
            for tok, te in ((self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)):
                ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True,
                          return_tensors="pt").input_ids.to(self.device)
                out = te(ids, output_hidden_states=True)
                embs.append(out.hidden_states[-2])
                if te is self.text_encoder_2:
                    pooled = out[0]
            emb = (torch.cat(embs, dim=-1).to("cpu"), pooled.to("cpu"))
        if len(self._emb_cache) < 4000:
            self._emb_cache[text] = emb
        return emb

    def _te_to(self, dev):
        self.text_encoder.to(dev)
        self.text_encoder_2.to(dev)

    def encode_text(self, text):
        if text in self._emb_cache:
            return self._emb_cache[text]
        self._te_to(self.device)
        try:
            return self._encode_text_on_device(text)
        finally:
            self._te_to("cpu")

    def encode_texts(self, texts, progress_cb=None):
        self._te_to(self.device)
        try:
            out = []
            for n, t in enumerate(texts):
                out.append(self._encode_text_on_device(t))
                if progress_cb:
                    progress_cb(n + 1, len(texts))
            return out
        finally:
            self._te_to("cpu")
            free_gpu_memory()

    def _added(self, pooled, n):
        torch = self.torch
        r = float(self.res)
        tid = torch.tensor([[r, r, 0.0, 0.0, r, r]] * n, device=self.device, dtype=self.wdtype)
        return {"text_embeds": pooled.to(self.device, self.wdtype), "time_ids": tid}

    def step_sft(self, lat, emb, noise=None, t=None, weight=None):
        torch = self.torch
        self._ensure_unet()
        prompt, pooled = emb
        lat = lat.to(self.device, self.wdtype)
        prompt = prompt.to(self.device, self.wdtype)
        if noise is None:
            noise = torch.randn_like(lat)
            if self.noise_offset:
                noise = noise + self.noise_offset * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=lat.device,
                                                                dtype=lat.dtype)
        if t is None:
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=lat.device)
        noisy = self.noise_scheduler.add_noise(lat, noise, t.long())
        with self._autocast():
            pred = self.unet(noisy, t, prompt, added_cond_kwargs=self._added(pooled, lat.shape[0])).sample
        target = self._target(lat, noise, t)
        if weight is None:
            loss = self._mse(pred, target)
        else:
            w = weight.to(self.device, torch.float32)
            err = (pred.float() - target.float()) ** 2
            loss = (err * w).sum() / (w.sum() * err.shape[1]).clamp_min(1e-6)
        self._backward(loss)
        self._opt_step()
        return float(loss.detach())

    def step_dpo(self, *a, **k):
        raise NotImplementedError("preference training is SD 1.5 only")

    def save(self, out_path, info):
        from diffusers import StableDiffusionXLPipeline
        d, fn = os.path.split(out_path)
        os.makedirs(d, exist_ok=True)
        check_space(d, 0.3)
        part = fn + ".part"
        StableDiffusionXLPipeline.save_lora_weights(save_directory=d, unet_lora_layers=self.lora_state(),
                                                    weight_name=part, safe_serialization=True)
        os.replace(os.path.join(d, part), out_path)
        with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        return out_path

    def sample(self, prompt, negative="", seed=1234, steps=20, cfg=7.0, size=None):
        torch = self.torch
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        sched = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config, use_karras_sigmas=True,
                                                        algorithm_type="dpmsolver++")
        self._ensure_unet()
        dev = self.device
        cls = type("AilsSampleXL", (StableDiffusionXLPipeline,), {"_execution_device": property(lambda _s: dev)})
        pipe = cls(vae=self.vae, text_encoder=self.text_encoder, text_encoder_2=self.text_encoder_2,
                   tokenizer=self.tokenizer, tokenizer_2=self.tokenizer_2, unet=self.unet, scheduler=sched)
        pipe.set_progress_bar_config(disable=True)
        size = int(size or self.res)
        self.vae.to(self.device)
        self._te_to(self.device)
        self.unet.eval()
        try:
            with torch.no_grad(), self._autocast():
                img = pipe(prompt, negative_prompt=negative or None, num_inference_steps=steps, guidance_scale=cfg,
                           width=size, height=size, generator=torch.Generator("cpu").manual_seed(int(seed)),
                           callback_on_step_end=self._abort_cb).images[0]
        finally:
            self.unet.train()
            self.vae.to("cpu")
            self._te_to("cpu")
            free_gpu_memory()
        return img

    def close(self):
        if self.swapper is not None:
            try:
                self.swapper.remove(to=None)
            except Exception:
                pass
        self.swapper = None
        self.unet = self.vae = self.text_encoder = self.text_encoder_2 = None
        self.opt = None
        free_gpu_memory()


class FlowLoraTrainer(LoraTrainer):
    """
    Akış eşleştirmeli (flow matching) DiT modelleri için LoRA eğitimi: FLUX.2 [klein] 4B ve Z-Image.
    Dönüştürücü base (damıtılmamış) modelden, VAE / metin kodlayıcı / sözlük kurulu (hızlı) modelden alınır.
    Bellek sırası: önce görseller (VAE), sonra metinler (metin kodlayıcı; iş bitince bellekten tamamen silinir),
    en son dönüştürücü GPU'ya alınır — 12 GB'ta FLUX.2 eğitimi böyle sığar.
    """

    def __init__(self, model_path, base_dir, model, device, rank=STYLE_RANK, lr=STYLE_LR, resolution=None):
        torch = import_torch()
        import contextlib
        import importlib
        import diffusers
        import transformers
        from peft import LoraConfig
        self.torch, self._ctx = torch, contextlib
        self.key = model["key"]
        self.flux = self.key == "flux2_klein_4b"
        self.device = torch.device(device)
        self.on_gpu = self.device.type == "cuda"
        self.wdtype = torch.bfloat16 if self.on_gpu else torch.float32
        self.res = int(resolution or TRAIN_SPECS[self.key]["res"])
        self.res = max(64, self.res // 16 * 16)
        with open(os.path.join(model_path, "model_index.json"), encoding="utf-8") as f:
            idx = json.load(f)

        def cls_of(comp):
            lib, name = idx[comp]
            return getattr(diffusers if lib == "diffusers" else transformers, name)

        kw = dict(torch_dtype=self.wdtype)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = cls_of("text_encoder").from_pretrained(model_path, subfolder="text_encoder", **kw)
        self.vae = cls_of("vae").from_pretrained(model_path, subfolder="vae", **kw)
        tdir = base_dir if os.path.isdir(os.path.join(base_dir, "transformer")) else model_path
        self.transformer = cls_of("transformer").from_pretrained(tdir, subfolder="transformer", **kw)
        self.noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(model_path,
                                                                                          subfolder="scheduler")
        self._pipe_cls = cls_of_pipe = getattr(diffusers, idx["_class_name"])
        self._pipe_mod = importlib.import_module(cls_of_pipe.__module__)
        for m in (self.text_encoder, self.vae, self.transformer):
            m.requires_grad_(False)
        if self.flux:
            targets = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                       "to_qkv_mlp_proj"]
        else:
            targets = ["to_q", "to_k", "to_v", "to_out.0"]
        self.lora_config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=targets)
        self.transformer.add_adapter(self.lora_config)
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.params = [p for p in self.transformer.parameters() if p.requires_grad]
        try:
            self.transformer.enable_gradient_checkpointing()
        except Exception:
            pass
        self.transformer.train()
        self._on_device = False
        self.swapper = None
        self.swap_info = None             # (ekran kartında GB, RAM'den taşınan GB) — RAM destekli ise
        self.reserve = int(2.0 * GB)      # ara sonuçlar + LoRA + iyileştirici durumu için boş bırakılan VRAM
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-2)
        self.scaler = None
        self.pred_type = "flow"
        self.steps = 0
        self._emb_cache = {}
        self.noise_offset = 0.0
        vs = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale = vs * 2 if self.flux else vs        # ağırlık haritası: FLUX.2'de 2×2 yamalı token ızgarası

    # ---- yardımcılar
    def _autocast(self):
        if self.on_gpu:
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return self._ctx.nullcontext()

    def _ensure_transformer(self):
        """Dönüştürücüyü ekran kartına alır; sığmıyorsa RAM destekli (blok takası) çalıştırır."""
        if self._on_device:
            return
        self._on_device = True
        if not self.on_gpu:
            self.transformer.to(self.device)
            return
        free_gpu_memory()
        need = module_bytes(self.transformer)
        free = gpu_free_bytes(self.device)
        if need + self.reserve <= free:
            self.transformer.to(self.device)
            return
        self.swapper = ParamSwapper(self.transformer, self.device, max(0, free - self.reserve),
                                    abort_event=getattr(self, "abort_event", None))
        self.swap_info = (self.swapper.resident_bytes / GB, self.swapper.swapped_bytes / GB)

    def prepare_device(self):
        self._ensure_transformer()
        return self.swap_info

    def _encode_latent(self, x):
        torch = self.torch
        x = x.to(self.device, self.wdtype)
        if self.flux:
            lat = self.vae.encode(x).latent_dist.mode()
            lat = self._pipe_cls._patchify_latents(lat)
            mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
            std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                lat.device, lat.dtype)
            return (lat - mean) / std
        lat = self.vae.encode(x).latent_dist.sample()
        return (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def _encode_text_on_device(self, text):
        torch = self.torch
        if text in self._emb_cache:
            return self._emb_cache[text]
        with torch.no_grad():
            if self.flux:
                msgs = [{"role": "user", "content": text}]
                t = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
                ti = self.tokenizer(t, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                out = self.text_encoder(input_ids=ti["input_ids"].to(self.device),
                                        attention_mask=ti["attention_mask"].to(self.device),
                                        output_hidden_states=True, use_cache=False)
                hs = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1)
                b, ch, ln, d = hs.shape
                emb = hs.permute(0, 2, 1, 3).reshape(b, ln, ch * d).to("cpu", self.wdtype)
            else:
                msgs = [{"role": "user", "content": text}]
                t = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=True)
                ti = self.tokenizer([t], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
                ids = ti.input_ids.to(self.device)
                mask = ti.attention_mask.to(self.device).bool()
                hs = self.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
                emb = hs[0][mask[0]].to("cpu", self.wdtype)
        if len(self._emb_cache) < 4000:
            self._emb_cache[text] = emb
        return emb

    def encode_text(self, text):
        if text in self._emb_cache:
            return self._emb_cache[text]
        if self.text_encoder is None:
            raise RuntimeError("text encoder already released")
        sw = self._te_on_device()
        try:
            return self._encode_text_on_device(text)
        finally:
            self._te_off(sw)

    def encode_texts(self, texts, progress_cb=None):
        """Tüm metinler (ve örnek görselin boş negatif metni) kodlanır; sonra metin kodlayıcı bellekten silinir."""
        sw = self._te_on_device()
        try:
            out = []
            ab = getattr(self, "abort_event", None)
            for n, t in enumerate(texts):
                if ab is not None and ab.is_set():
                    raise AbortedError("user")
                out.append(self._encode_text_on_device(t))
                if progress_cb:
                    progress_cb(n + 1, len(texts))
            self._encode_text_on_device("")
            return out
        finally:
            self._te_off(sw)
            self.text_encoder = None
            free_gpu_memory()

    def _te_on_device(self):
        """Metin kodlayıcı ekran kartına; sığmıyorsa RAM destekli. Dönüş: takasçı ya da None."""
        if not self.on_gpu:
            return None
        free_gpu_memory()
        need = module_bytes(self.text_encoder)
        free = gpu_free_bytes(self.device)
        if need + int(1.0 * GB) <= free:
            self.text_encoder.to(self.device)
            return None
        # metin kodlayıcı yalnızca bir kez, kısa süre çalışır: sabitlenmiş bellek kullanılmaz (RAM'i kilitlemesin)
        return ParamSwapper(self.text_encoder, self.device, max(0, free - int(1.5 * GB)), pin=False,
                            abort_event=getattr(self, "abort_event", None))

    def _te_off(self, sw):
        if sw is not None:
            sw.remove(to="cpu")
        elif self.text_encoder is not None:
            self.text_encoder.to("cpu")
        free_gpu_memory()

    def _sigmas(self, n, seq_len):
        torch = self.torch
        u = torch.sigmoid(torch.randn(n, device=self.device))
        if self.flux:
            mu = self._pipe_mod.compute_empirical_mu(image_seq_len=seq_len, num_steps=50)
        else:
            mu = self._pipe_mod.calculate_shift(seq_len)
        e = math.exp(mu)
        return (e / (e + (1.0 / u.clamp(1e-4, 1 - 1e-4) - 1.0))).clamp(1e-3, 1.0)

    def _predict(self, noisy, sigma, emb):
        """Modelin hız tahmini (v = gürültü − görsel), (B, C, H, W) biçiminde."""
        torch = self.torch
        if self.flux:
            pc = self._pipe_cls
            tokens = pc._pack_latents(noisy)
            img_ids = pc._prepare_latent_ids(noisy).to(self.device)
            txt = emb.to(self.device, self.wdtype)
            txt_ids = pc._prepare_text_ids(txt).to(self.device)
            out = self.transformer(hidden_states=tokens.to(self.wdtype), timestep=sigma.to(self.wdtype), guidance=None,
                                   encoder_hidden_states=txt, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
            b, c, h, w = noisy.shape
            return out.permute(0, 2, 1).reshape(b, c, h, w)
        x = [noisy[0].unsqueeze(1).to(self.wdtype)]
        cap = [emb.to(self.device, self.wdtype)]
        out = self.transformer(x, (1.0 - sigma).to(self.wdtype), cap, return_dict=False)[0]
        return -torch.stack([o.float() for o in out], dim=0).squeeze(2)

    def step_sft(self, lat, emb, noise=None, t=None, weight=None):
        torch = self.torch
        self._ensure_transformer()
        x0 = lat.to(self.device, torch.float32)
        if noise is None:
            noise = torch.randn_like(x0)
        seq = (x0.shape[2] * x0.shape[3]) if self.flux else (x0.shape[2] // 2) * (x0.shape[3] // 2)
        sigma = self._sigmas(x0.shape[0], seq) if t is None else t.to(self.device, torch.float32)
        s = sigma.view(-1, 1, 1, 1)
        noisy = (1.0 - s) * x0 + s * noise
        with self._autocast():
            pred = self._predict(noisy, sigma, emb)
        target = noise - x0
        if weight is None:
            loss = self._mse(pred, target)
        else:
            w = weight.to(self.device, torch.float32)
            if w.shape[-2:] != pred.shape[-2:]:
                w = torch.nn.functional.interpolate(w, size=pred.shape[-2:], mode="nearest")
            err = (pred.float() - target.float()) ** 2
            loss = (err * w).sum() / (w.sum() * err.shape[1]).clamp_min(1e-6)
        loss.backward()
        self._opt_step()
        return float(loss.detach())

    def step_dpo(self, *a, **k):
        raise NotImplementedError("preference training is SD 1.5 only")

    # ---- kaydet / devam et / örnek görsel
    def lora_state(self):
        from peft.utils import get_peft_model_state_dict
        return get_peft_model_state_dict(self.transformer)

    def save(self, out_path, info):
        d, fn = os.path.split(out_path)
        os.makedirs(d, exist_ok=True)
        check_space(d, 0.5)
        part = fn + ".part"
        self._pipe_cls.save_lora_weights(save_directory=d, transformer_lora_layers=self.lora_state(),
                                         weight_name=part, safe_serialization=True)
        os.replace(os.path.join(d, part), out_path)
        with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        return out_path

    def load_state(self, path):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(path)
        sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        res = set_peft_model_state_dict(self.transformer, sd, adapter_name="default")
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return res

    def sample(self, prompt, negative="", seed=1234, steps=None, cfg=None, size=None):
        torch = self.torch
        sp = TRAIN_SPECS[self.key]
        steps = int(steps or sp["sample_steps"])
        cfg = float(sp["sample_cfg"] if cfg is None else cfg)
        pos = self.encode_text(prompt)
        neg = self.encode_text(negative or "")
        self._ensure_transformer()
        kwargs = dict(scheduler=self.noise_scheduler, vae=self.vae, text_encoder=None, tokenizer=self.tokenizer,
                      transformer=self.transformer)
        if self.flux:
            kwargs["is_distilled"] = False
        dev = self.device
        cls = type("AilsSamplePipe", (self._pipe_cls,), {"_execution_device": property(lambda _s: dev)})
        pipe = cls(**kwargs)
        pipe.set_progress_bar_config(disable=True)
        size = int(size or self.res)
        self.vae.to(self.device)
        self.transformer.eval()
        g = torch.Generator("cpu").manual_seed(int(seed))
        ab = getattr(self, "abort_event", None)

        def _cb(_p, _i, _t, kw):
            if ab is not None and ab.is_set():
                raise AbortedError("user")
            return kw
        try:
            with torch.no_grad(), self._autocast():
                if self.flux:
                    img = pipe(prompt_embeds=pos.to(self.device, self.wdtype),
                               negative_prompt_embeds=neg.to(self.device, self.wdtype), num_inference_steps=steps,
                               guidance_scale=cfg, width=size, height=size, generator=g,
                               callback_on_step_end=_cb).images[0]
                else:
                    img = pipe(prompt_embeds=[pos.to(self.device, self.wdtype)],
                               negative_prompt_embeds=[neg.to(self.device, self.wdtype)], num_inference_steps=steps,
                               guidance_scale=cfg, width=size, height=size, generator=g,
                               callback_on_step_end=_cb).images[0]
        finally:
            self.transformer.train()
            self.vae.to("cpu")
            free_gpu_memory()
        return img

    def close(self):
        if self.swapper is not None:
            try:
                self.swapper.remove(to=None)
            except Exception:
                pass
        self.swapper = None
        self.transformer = self.vae = self.text_encoder = None
        self.opt = None
        free_gpu_memory()


class NothingToTrainError(Exception):
    """Eğitilecek veri yok (ör. hiç puan verilmemiş ve otomatik hakem kapalı)."""


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


class _TrainClock:
    """Süre sınırı, ara kayıt zamanı ve durdurma kontrolü."""

    def __init__(self, minutes, checkpoint_min, abort_event, clock=time.time):
        self.clock = clock
        self.start = clock()
        self.limit = float(minutes) * 60.0 if minutes else None
        self.ck = max(1.0, float(checkpoint_min or 15)) * 60.0
        self.last_ck = self.start
        self.abort = abort_event

    def elapsed(self):
        return self.clock() - self.start

    def should_stop(self):
        if self.abort is not None and self.abort.is_set():
            return "stop"
        if self.limit is not None and self.elapsed() >= self.limit:
            return "time"
        return None

    def checkpoint_due(self):
        if self.clock() - self.last_ck >= self.ck:
            self.last_ck = self.clock()
            return True
        return False

    def fraction(self):
        return min(1.0, self.elapsed() / self.limit) if self.limit else None


def _train_progress(progress_cb, clock, steps):
    if progress_cb:
        f = clock.fraction()
        progress_cb(f, steps, clock.elapsed())


def run_style_training(paths, hw, slug, tag, minutes=None, checkpoint_min=15, sample_each=True, log_cb=None, msg=None,
                       progress_cb=None, abort_event=None, on_sample=None, model_key="sd15", resolution=None,
                       max_images=1000, clock=time.time, sample_steps=20, learn=None, hide_faces=False):
    """
    Derin mod: veri setindeki görsellerle bir LoRA eğitir. Ne öğretildiği (learn: stil / mekân / konu / anatomi)
    veri setinden okunur ve eğitim ayarlarını belirler. Durdurulana ya da süre dolana kadar sürer; her ara kayıtta
    LoRA kaydedilir (istenirse örnek görsel üretilir). Aynı LoRA yeniden eğitilirse kaldığı yerden devam eder;
    başka bir eğitime ait LoRA asla ezilmez (yeni ad alınır).
    hide_faces (yalnızca mekân): yüzler bulanıklaştırılır ve öğretilmez, insanlar az öğretilir — çevre öne çıkar.
    progress_cb(kesir|None, adım, geçen_sn). Dönüş: bilgi sözlüğü.
    """
    log = log_cb or (lambda s: None)
    msg = msg or (lambda k, **kw: k)
    model = IMAGE_MODEL_BY_KEY[model_key]
    if model_key not in TRAIN_BASES:
        raise RuntimeError("training is only supported for: " + ", ".join(TRAIN_BASES))
    data = load_dataset(paths, slug)
    if not data or not data["items"]:
        raise NothingToTrainError("empty dataset")
    learn = norm_learn(learn or data.get("learn"))
    preset = LEARN_PRESETS[learn]
    tag = (tag or data.get("tag") or data.get("name") or slug).strip()
    trigger = ascii_trigger(tag, learn)
    base_name = safe_folder_name(tag, learn) + ("" if model_key == "sd15" else "_" + TRAIN_SUFFIX[model_key])
    lora_dir = own_lora_dir(paths)
    os.makedirs(lora_dir, exist_ok=True)
    out, prev = pick_lora_path(lora_dir, base_name, slug, learn, preset["rank"], base=model_key)
    name = os.path.splitext(os.path.basename(out))[0]
    hide = bool(hide_faces) and learn == "place"
    face_mode = learn == "face"
    if (hide or face_mode) and not (detector_files_ok(paths.models) and cv2_installed(paths)):
        raise RuntimeError("face detector is not installed (needed to " + ("find faces)" if face_mode else
                                                                           "hide faces)"))
    ok, _why = train_ready(hw, model_key)
    if not ok:
        raise TrainVramError((model_key, TRAIN_SPECS[model_key]["vram"], hw.vram_gb if hw.cuda_usable else 0.0))
    if TRAIN_SPECS[model_key].get("base_repo") and not is_train_base_installed(paths.models, model_key):
        raise RuntimeError("training base model is not installed: " + model_key)
    device = train_device(hw) if model_key == "sd15" else hw.device
    log(msg("tr_log_load", device="GPU" if device != "cpu" else "CPU"))
    log(msg("tr_log_kind", kind=msg("learn_" + learn)))
    log(msg("tr_log_base", base=model["name"]))
    if name != base_name and os.path.exists(os.path.join(lora_dir, base_name + ".safetensors")):
        log(msg("tr_log_keep_old", old=base_name + ".safetensors", new=name + ".safetensors"))
    trainer = make_trainer(paths, model_key, device, rank=preset["rank"], lr=preset["lr"], resolution=resolution)
    trainer.abort_event = abort_event
    trainer.noise_offset = preset["noise_offset"]
    if prev:
        try:
            trainer.load_state(out)
            log(msg("tr_log_resume", steps=prev.get("steps", 0)))
        except Exception:
            prev = {}
    from PIL import Image
    entries, texts = [], []
    n_img = n_faces = n_with = n_noface = 0
    for it in data["items"][:max_images]:
        if abort_event is not None and abort_event.is_set():
            raise AbortedError("user")
        p = dataset_item_path(paths, slug, it)
        try:
            with Image.open(p) as im:
                w, h = im.size
                people = find_people(paths.models, im) if (hide or face_mode) else None
        except Exception:
            continue
        if face_mode:
            big = [b for b in (people or []) if max(b[2], b[3]) >= FACE_MIN_PX]
            if not big:
                n_noface += 1
                continue
            n_img += 1
            entries.append((p, "face", [max(big, key=lambda b: b[2] * b[3])]))
            texts.append(trigger + ", " + FACE_PROMPT)
            continue
        n_img += 1
        if people:
            n_faces += len(people)
            n_with += 1
        text = ", ".join(x for x in (trigger, it.get("caption", "")) if x)
        for v in image_views(w, h, preset["views"]):
            entries.append((p, v, people or []) if hide else (p, v))
            texts.append(text)
    if face_mode:
        log(msg("tr_log_face_mode", n=n_img, skipped=n_noface))
    if not entries:
        raise NothingToTrainError("no readable images")
    log(msg("tr_log_encode", n=n_img) + ("" if len(entries) == n_img else "  " + msg("tr_log_views", v=len(entries))))
    if hide:
        log(msg("tr_log_faces", faces=n_faces, images=n_with))
    lat_all, w_all = trainer.encode_images(entries, abort_event=abort_event, skip_bad=True, with_weights=True)
    # neredeyse tamamı insan olan kesitler atlanır (öğretilecek çevre kalmamış)
    keep = [i for i, lat in enumerate(lat_all)
            if lat is not None and (w_all[i] is None or float(w_all[i].float().mean()) >= 0.05)]
    if not keep:
        raise NothingToTrainError("no readable images")
    latents = [lat_all[i] for i in keep]
    weights = [w_all[i] for i in keep]
    texts = [texts[i] for i in keep]
    del lat_all, w_all
    embs = trainer.encode_texts(texts)
    sample_prompt = texts[0]
    if hasattr(trainer, "prepare_device"):
        sw = trainer.prepare_device()
        if sw:
            log(msg("tr_log_swap", gpu=sw[0], ram=sw[1]))
    base_steps = int(prev.get("steps", 0))
    info = {"name": tag, "tag": tag, "trigger": trigger, "base": model_key, "type": learn, "learn": learn,
            "dataset": slug, "images": n_img, "views": len(latents), "rank": preset["rank"], "res": trainer.res,
            "faces_hidden": hide, "face_only": face_mode,
            "created": prev.get("created") or time.strftime("%Y-%m-%d %H:%M")}
    ck = _TrainClock(minutes, checkpoint_min, abort_event, clock)
    import random as _r
    losses = []
    log(msg("tr_log_train", trigger=trigger))

    def checkpoint(final=False, stopped=False):
        info.update(steps=base_steps + trainer.steps, updated=time.strftime("%Y-%m-%d %H:%M"),
                    loss=round(sum(losses[-200:]) / max(1, len(losses[-200:])), 5) if losses else None)
        if sample_each and not stopped:
            try:
                d = os.path.join(own_lora_dir(paths), name + "_samples")
                os.makedirs(d, exist_ok=True)
                sp = os.path.join(d, "step_%06d.png" % info["steps"])
                trainer.sample(sample_prompt, seed=1234, steps=sample_steps).save(sp)
                info["sample"] = sp
                if on_sample:
                    on_sample(sp, info)
            except AbortedError:
                stopped = True              # örnek görsel yarıda kaldı: LoRA yine de kaydedilir
            except Exception as e:
                if is_oom_error(e):
                    log(msg("tr_log_sample_oom"))
                else:
                    raise
        trainer.save(out, info)
        log(msg("tr_log_saved_final" if final else "tr_log_saved", steps=info["steps"], loss=info.get("loss") or 0.0))
        return stopped

    reason = None
    last_report = [0.0]

    def report(force=False):
        now = time.monotonic()
        if force or now - last_report[0] >= 1.0:       # adım ne kadar hızlı olursa olsun en çok saniyede bir
            last_report[0] = now
            _train_progress(progress_cb, ck, base_steps + trainer.steps)

    try:
        report(force=True)
        while True:
            reason = ck.should_stop()
            if reason:
                break
            i = _r.randrange(len(latents))
            try:
                losses.append(trainer.step_sft(latents[i], embs[i], weight=weights[i]))
            except AbortedError:
                try:
                    trainer.opt.zero_grad(set_to_none=True)     # yarım kalan adım öğretilmez
                except Exception:
                    pass
                reason = "stop"
                break
            report()
            if trainer.steps % 200 == 0:
                log(msg("tr_log_step", steps=base_steps + trainer.steps,
                        loss=sum(losses[-200:]) / len(losses[-200:])))
            if ck.checkpoint_due():
                if checkpoint():
                    reason = "stop"
                    break
        report(force=True)
        if reason == "stop":
            log(msg("tr_log_stopping"))
        checkpoint(final=True, stopped=(reason == "stop"))
    finally:
        trainer.close()
    info["stop_reason"] = reason
    info["path"] = out
    return info


def run_preference_training(paths, hw, minutes=None, checkpoint_min=15, sample_each=True, auto_judge=True,
                            prompt_pool=None, log_cb=None, msg=None, progress_cb=None, abort_event=None, on_sample=None,
                            resolution=None, clock=time.time, gen_steps=20, judge_margin=0.05):
    """
    Kendi kendine öğrenme (ödül/ceza): 👍 = ödül, 👎 = ceza. Aynı prompt için 👍 ve 👎 alan görseller
    Diffusion-DPO ile karşılaştırılır; tek kalan 👍'ler normal eğitimle pekiştirilir. Otomatik hakem açıksa
    program aynı prompt'la iki görsel üretir, hakem puanına göre iyisini ödüllendirip kötüsünü cezalandırır.
    Sonuç her seferinde birikerek 'ails_tercih_sd15' LoRA'sına kaydedilir.
    """
    log = log_cb or (lambda s: None)
    msg = msg or (lambda k, **kw: k)
    model = IMAGE_MODEL_BY_KEY["sd15"]
    pairs, singles, stats = feedback_training_data(paths, "sd15")
    if not pairs and not singles and not auto_judge:
        raise NothingToTrainError("no ratings")
    device = train_device(hw)
    log(msg("tr_log_load", device="GPU" if device != "cpu" else "CPU"))
    trainer = LoraTrainer(image_model_dir(paths.models, model), model, device, rank=PREF_RANK, lr=PREF_LR,
                          resolution=resolution)
    out = os.path.join(own_lora_dir(paths), PREF_LORA_NAME + ".safetensors")
    prev = {}
    if os.path.exists(out):
        try:
            with open(os.path.splitext(out)[0] + ".json", encoding="utf-8") as f:
                prev = json.load(f)
            trainer.load_state(out)
            log(msg("tr_log_resume", steps=prev.get("steps", 0)))
        except Exception:
            prev = {}
    user_pairs, user_singles = [], []
    for g, b in pairs:
        try:
            lw, ll = trainer.encode_images([_open_rgb(g["path"]), _open_rgb(b["path"])])
            user_pairs.append((lw, ll, trainer.encode_text(g.get("prompt") or "")))
        except Exception:
            continue
    for s in singles:
        try:
            user_singles.append((trainer.encode_images([_open_rgb(s["path"])])[0], trainer.encode_text(s.get("prompt") or "")))
        except Exception:
            continue
    pool = [p for p in (prompt_pool or []) if p] or list(DEFAULT_SELF_PROMPTS)
    log(msg("tr_log_pref", pairs=len(user_pairs), singles=len(user_singles),
            judge=msg("on") if auto_judge else msg("off")))
    base_steps = int(prev.get("steps", 0))
    counters = {"judged": int(prev.get("judged", 0)), "user_steps": 0, "acc": []}
    info = {"name": "AILS tercih (SD 1.5)", "tag": "tercih", "trigger": "", "base": "sd15", "type": "preference",
            "rank": PREF_RANK, "res": trainer.res, "created": prev.get("created") or time.strftime("%Y-%m-%d %H:%M")}
    ck = _TrainClock(minutes, checkpoint_min, abort_event, clock)
    import random as _r

    def checkpoint(final=False):
        acc = counters["acc"][-100:]
        info.update(steps=base_steps + trainer.steps, judged=counters["judged"], pairs=len(user_pairs),
                    accuracy=round(sum(acc) / len(acc), 3) if acc else None, updated=time.strftime("%Y-%m-%d %H:%M"))
        if sample_each:
            try:
                d = os.path.join(own_lora_dir(paths), PREF_LORA_NAME + "_samples")
                os.makedirs(d, exist_ok=True)
                sp = os.path.join(d, "step_%06d.png" % info["steps"])
                trainer.sample(pool[0], seed=777, steps=gen_steps).save(sp)
                info["sample"] = sp
                if on_sample:
                    on_sample(sp, info)
            except Exception as e:
                if is_oom_error(e):
                    log(msg("tr_log_sample_oom"))
                else:
                    raise
        trainer.save(out, info)
        log(msg("tr_log_pref_saved", steps=info["steps"], judged=info["judged"],
                acc=(info["accuracy"] if info["accuracy"] is not None else 0.0) * 100.0))

    k = 0
    reason = None
    try:
        while True:
            reason = ck.should_stop()
            if reason:
                break
            did = False
            if user_pairs:
                w, lo, e = user_pairs[k % len(user_pairs)]
                _l, a = trainer.step_dpo(w, lo, e)
                counters["acc"].append(a)
                did = True
            if user_singles and k % 3 == 0:
                lat, e = user_singles[(k // 3) % len(user_singles)]
                trainer.step_sft(lat, e)
                did = True
            if auto_judge and (k % 2 == 0 or not did):
                prompt = _r.choice(pool)
                s1, s2 = _r.randrange(1, 2 ** 31 - 1), _r.randrange(1, 2 ** 31 - 1)
                ia = trainer.sample(prompt, seed=s1, steps=gen_steps)
                if ck.should_stop():
                    reason = ck.should_stop()
                    break
                ib = trainer.sample(prompt, seed=s2, steps=gen_steps)
                sa, _pa = judge_image(paths.models, ia, prompt)
                sb, _pb = judge_image(paths.models, ib, prompt)
                counters["judged"] += 1
                if abs(sa - sb) >= judge_margin:
                    win, lose = (ia, ib) if sa > sb else (ib, ia)
                    lw, ll = trainer.encode_images([win, lose])
                    _l, a = trainer.step_dpo(lw, ll, trainer.encode_text(prompt))
                    counters["acc"].append(a)
            k += 1
            _train_progress(progress_cb, ck, base_steps + trainer.steps)
            if ck.checkpoint_due():
                checkpoint()
        checkpoint(final=True)
    finally:
        trainer.close()
    info["stop_reason"] = reason
    info["path"] = out
    return info
