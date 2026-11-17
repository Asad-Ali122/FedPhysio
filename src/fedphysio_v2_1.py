# %% [markdown]
# # FedPhysio v2 — the proposed method only
# ### *FedPhysio: XAI-Guided Modality Selection and Reliability-Aware Federated Learning for Multimodal Digital Health*
#
# This notebook trains and evaluates **only the proposed method**. The earlier run already produced the
# baselines and ablations. Everything here is designed to raise performance **without leakage**: the held-out
# subject's labels are never used for any choice, except inside the clearly separated *subject-calibrated*
# protocol, where a few labelled minutes of the new user are part of the protocol by definition.
#
# **What changed relative to the first FedPhysio (and why)**
#
# | # | Change in FedPhysio v2 | Problem it addresses | Evidence |
# |---|---|---|---|
# | 1 | **TRACE-H**: TRACE on raw 10-s windows **fused with a physiological feature branch** (HR/HRV, tonic/phasic EDA, SCR rate, respiration rate and variability, EMG, temperature slope, motion) computed over trailing **20-s and 60-s** context | Raw 10-s windows miss slow autonomic changes; Amusement was confused with Stress | Best honest WESAD LOSO results use physiological features and 60-s context (Schmidt 2018) |
# | 2 | **Robust per-subject normalisation** (median / IQR, label-free) for raw signals and features | S10-type subjects whose baseline looks like stress | Subject-wise normalisation is the most consistent lever in the literature |
# | 3 | **GroupNorm / LayerNorm instead of BatchNorm** (no batch statistics at all) | Run 1 showed that batch statistics estimated from class-balanced training batches do not match the natural class mix of a new subject (S2 collapsed) | Hsieh et al., ICML 2020 (*The non-IID data quagmire*): GroupNorm for federated learning |
# | 3b | **Clock-feature rule**: features whose mean \|Spearman ρ\| with time-in-session is ≥ 0.5 in the fold's **training subjects** are dropped (label-free, per fold) | Session-normalised levels (temperature, skin-conductance level) act as a clock that encodes the protocol order, which differs between subjects | Label-free check; WESAD "clock" shortcut |
# | 3c | **Validation-tuned prior correction**: training uses class-balanced sampling; at prediction time, τ_p·log π is added to the logits (π = class prior from the summed client validation counts). τ_p ∈ {0, 0.5, 1} is chosen per run on the **validation Cohen's κ** of the training clients | Balanced sampling shifts the decision prior away from the natural class mix | Menon et al., ICLR 2021 (logit adjustment) |
# | 4 | **PRISM-RM aggregation**: the 1-bit per-class reliability flag (EMA-smoothed, 3 warm-up rounds) sets the weights of a **reliability-weighted coordinate-wise median** | Your run showed coordinate-wise Median was the strongest aggregator and the reliability bit is the best screen | Yin et al. 2018; earlier FedPhysio results |
# | 5 | **Minority-class data augmentation**: denser training windows for Amusement (stride 1 s) and Stress (2.5 s), class-balanced sampling, and class-dependent augmentation probability (more for rarer classes): within-client same-class mixing (TIA), channel-group recombination (PCR, raw and features together), magnitude scaling and jitter, feature jitter, channel masking | Amusement recall was 0.31 | Standard imbalance practice; all mixing stays inside one client |
# | 6 | **Snapshot ensemble** of the 3 best validation checkpoints | Round-to-round variance | Pre-specified |
# | 7 | **Sensor subsets**: the 8 candidate subsets of the paper are all trained; the subset of each fold is chosen on **internal validation only** | Requested; the choice must not look at the test subject | — |
#
# **Protocols reported (each in its own table)**
# 1. **Strict LOSO**: window-level, the primary result.
# 2. **Strict LOSO + causal 30-s smoothing**: trailing average of the predicted probabilities over the past 30 s. It only uses past model outputs, but WESAD's long condition blocks favour smoothing, so report it as a separate line.
# 3. **Subject-calibrated**: the model is adapted with the **first 60 s / 120 s of each condition** of the new subject (labelled). The next 60 s are a buffer that is never tested, and evaluation uses only later windows. Strict LOSO on exactly the same windows is reported next to it.
#
# **Scenarios**: good (all clients, clean), average (50% participation, 2 corrupted clients, noisy test signals) and worst (30% participation, 4 corrupted clients, heavier test noise and sensor drop). They run with the fold's selected subset, and the screening rate of the reliability bit is reported for each.
#
# **Honest expectation (read before running).** No leakage-free 3-class WESAD LOSO result in the literature clearly exceeds about 85–89% accuracy. Reported 95–99% values come from subject-overlapping or within-subject splits. Under strict LOSO, aim for about **84–88%**. **>95% is only realistic under the subject-calibrated protocol**, so report it separately and name it as such.
#
# **Runtime (RTX 3090, one seed):** 8 subsets + 2 scenarios = 10 federated runs per fold, 150 in total, at about 8–17 min each. Cell 0 runs **3 workers on GPU 3 automatically**: expect roughly 10–12 h.
#
# **Crash-safe:** every finished run is cached, and an interrupted run resumes from its last communication round.
# Set `QUICK=True` for a short smoke test first.

# %%
# Installation (once, in your conda env, Tsinghua mirror)
# !pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy pandas scikit-learn matplotlib nbformat

# %% [markdown]
# ## 1. Configuration (all settings are fixed before any result is seen)

# %% [markdown]
# ### 0. One-click parallel run: 3 workers on GPU 3
# Running this notebook makes **this kernel worker 0/3** and starts **workers 1/3 and 2/3** in the background,
# all on GPU 3. The background workers run the companion script `fedphysio_v2_1.py`, which must be in the same
# folder. The three workers split the folds, then wait for each other, and every worker writes the same final tables.
# Background progress is written to `worker1.log` and `worker2.log` (`tail -f worker1.log`).
# Running the notebook again does **not** start duplicate workers while the previous ones are still alive.
# Finished runs are cached, and interrupted runs resume from their last round.

# %%
import os, sys, subprocess, json as _json
from pathlib import Path as _Path
N_WORKERS = 3            # parallel workers on one GPU (each needs ~2-4 GB)
GPU_ID = '3'             # the free GPU on your server
SCRIPT = 'fedphysio_v2_1.py'
_lock = _Path('fedphysio_v2_results') / 'logs' / 'workers.json'; _lock.parent.mkdir(parents=True, exist_ok=True)
def _alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False
if 'FEDPHYSIO_WORKER' not in os.environ:                       # only the interactive notebook launches workers
    os.environ['FEDPHYSIO_GPU'] = GPU_ID
    os.environ['FEDPHYSIO_WORKER'] = f'0/{N_WORKERS}'
    old = _json.load(open(_lock)) if _lock.exists() else {}
    pids = {}
    for w in range(1, N_WORKERS):
        if str(w) in old and _alive(old[str(w)]):
            print(f'worker {w}/{N_WORKERS} already running (PID {old[str(w)]})'); pids[str(w)] = old[str(w)]; continue
        if not _Path(SCRIPT).exists():
            raise FileNotFoundError(f'{SCRIPT} must be in the same folder as this notebook')
        env = dict(os.environ, FEDPHYSIO_WORKER=f'{w}/{N_WORKERS}', FEDPHYSIO_GPU=GPU_ID, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
        pr = subprocess.Popen([sys.executable, SCRIPT], env=env, cwd=os.getcwd(), start_new_session=True,
                              stdout=open(f'worker{w}.log', 'a'), stderr=subprocess.STDOUT)
        pids[str(w)] = pr.pid; print(f'started worker {w}/{N_WORKERS} on GPU {GPU_ID} (PID {pr.pid}) -> worker{w}.log')
    _json.dump(pids, open(_lock, 'w'))
print('this process:', os.environ['FEDPHYSIO_WORKER'], 'on GPU', os.environ.get('FEDPHYSIO_GPU'))

# %%
CFG = dict(
    DATA_ROOT = '/home/settings/dataset/WESAD',
    SUBJECTS  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    OUT       = 'fedphysio_v2_results',
    QUICK     = False,          # True = 3 folds, 2 subsets, few rounds (smoke test only)

    # signals and windows
    FS = 70, WIN = 700,                          # 10-s windows at 70 Hz
    STRIDE_TRAIN = {0: 350, 1: 175, 2: 70},      # TRAIN only: Baseline 5 s, Stress 2.5 s, Amusement 1 s (minority oversampling with real data)
    STRIDE_EVAL = 350,                           # validation and test: ONE label-independent stride (5 s)
    PURITY = 0.80, VAL_FRAC = 0.10,
    FEAT_CONTEXT_S = [20, 60],                   # trailing context of the physiological feature branch (seconds)
    CLIP_Z = 8.0,                                # clip robust z-scores

    # sensor subsets (paper candidates); the subset of each fold is chosen on internal validation
    SUBSETS = [['EDA', 'ECG', 'EMG', 'Resp'], ['EDA', 'ECG', 'Resp', 'Temp'], ['EDA', 'ECG', 'Temp', 'ACC'],
               ['EDA', 'ECG', 'Temp'], ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'],
               ['ECG', 'EDA', 'EMG', 'Resp', 'Temp'], ['EDA', 'ECG', 'ACC'], ['EDA', 'ECG', 'Resp']],

    # federated training
    R_MAX = 50, LOCAL_EPOCHS = 5, BATCH = 64, LR = 1e-3, WD = 1e-4, LABEL_SMOOTH = 0.05, PATIENCE = 10,
    TAU = 0.50, PENALTY = 0.3, EMA = 0.5, WARMUP = 3,       # PRISM reliability bit (fixed, not tuned)
    AGG = 'prism_wmedian',                       # reliability-weighted coordinate-wise median ('prism_mean' = weighted mean)
    FEDBN = False,                               # not needed: the model has no BatchNorm (GroupNorm / LayerNorm)
    ADABN = False,
    CLOCK_RHO = 0.5,                             # drop features with mean |Spearman rho| with time >= 0.5 in the TRAINING subjects
    PRIOR_TAUS = [0.0, 0.5, 1.0],                # prior correction tau_p, chosen on validation kappa (never on test)
    N_SNAPSHOT = 3,                              # ensemble of the 3 best validation checkpoints
    MOD_DROP = (0.10, 0.15),                     # modality dropout during training: (raw branch, feature branch)

    # minority-class augmentation (within client, same class)
    AUG = dict(base_p=0.5, max_p=0.95, p_mix=0.5, lam_lo=0.6, lam_hi=1.0, p_pcr=0.4, p_group=0.5,
               scale=0.10, jitter=0.03, feat_jitter=0.05, p_mask=0.10),

    # protocols
    SMOOTH_S = 30,                               # causal smoothing window (seconds)
    CAL_S = [60, 120], CAL_BUFFER_S = 60,        # calibrated protocol: labelled seconds per condition, buffer
    CAL_EPOCHS = 40, CAL_LR = 3e-4,

    # scenarios (fixed in advance)
    SCENARIOS = {
        'good':    dict(participation=1.0, n_corrupt=0, label_noise=0.0, sensor_noise=0.0, zero_group=False, test_noise=0.0, test_group_drop=0.0),
        'average': dict(participation=0.5, n_corrupt=2, label_noise=0.3, sensor_noise=0.5, zero_group=False, test_noise=0.1, test_group_drop=0.10),
        'worst':   dict(participation=0.3, n_corrupt=4, label_noise=0.5, sensor_noise=1.0, zero_group=True,  test_noise=0.3, test_group_drop=0.25),
    },
    SEEDS = [42],                # one seed (time budget); every SD reported is BETWEEN SUBJECTS
    AMP = False,                 # bf16 GRU is not supported by PyTorch 1.13 -> FP32
    DATA_VERSION = 1,
    RUN_VERSION = 2,             # v2.1: GroupNorm, clock rule, validation-tuned prior (old v2.0 runs are ignored)
    WORKER = os.environ.get('FEDPHYSIO_WORKER', '0/1'),
    REF_SUBMITTED = dict(acc=0.7831, kappa=0.6355, f1=0.6857),   # corrected FedPhysio (v1) result, for the summary only
)
if CFG['QUICK']:
    CFG.update(R_MAX=4, LOCAL_EPOCHS=1, PATIENCE=2, SUBSETS=CFG['SUBSETS'][:2], CAL_S=[30], CAL_BUFFER_S=10, CAL_EPOCHS=3)
FOLDS = CFG['SUBJECTS'][:3] if CFG['QUICK'] else CFG['SUBJECTS']

# %% [markdown]
# ## 2. Imports, device, utilities

# %%
import json, time, math, pickle, random, warnings, hashlib, zlib, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal as sps
from scipy.stats import skew
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, matthews_corrcoef,
                             roc_auc_score, precision_recall_fscore_support)
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def pick_device():
    gpu_id = int(os.environ.get('FEDPHYSIO_GPU', '3'))
    print(f'PyTorch {torch.__version__} | CUDA build: {torch.version.cuda}')
    if torch.version.cuda is None or not torch.cuda.is_available():
        print('!! No usable GPU -> CPU (slow).'); return torch.device('cpu')
    dev = torch.device(f'cuda:{min(gpu_id, torch.cuda.device_count() - 1)}')
    try:
        torch.cuda.set_device(dev)
        g = torch.nn.GRU(4, 8, batch_first=True).to(dev); g(torch.randn(4, 50, 4, device=dev))[0].sum().backward()
        print('Using', torch.cuda.get_device_name(dev)); return dev
    except Exception as e:
        print('!! GPU test failed:', e, '-> CPU'); return torch.device('cpu')

DEV = pick_device()
if DEV.type == 'cuda':
    free, total = torch.cuda.mem_get_info(DEV)
    print(f'GPU memory free: {free / 2**30:.1f} of {total / 2**30:.1f} GiB')
    if free < 4 * 2**30:
        print('!! Less than 4 GiB free: another process (often an old notebook kernel) is holding the GPU.\n'
              '   Run `nvidia-smi`, then shut down old kernels (Kernel -> Shut Down) or `kill <PID>` before continuing.')
torch.backends.cudnn.benchmark = True
OUT = Path(CFG['OUT'])
for d in ['cache', 'tables', 'figures', 'runs', 'logs', 'ckpt', 'states']:
    (OUT / d).mkdir(parents=True, exist_ok=True)
K = 3
CLASSES = ['Baseline', 'Stress', 'Amusement']
CH = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
GROUPS = {'ACC': [0, 1, 2], 'ECG': [3], 'EDA': [4], 'EMG': [5], 'Resp': [6], 'Temp': [7]}
GNAMES = list(GROUPS)
LAB = list(range(K))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def gkey(groups): return '+'.join(g for g in GNAMES if g in groups)
def chans(groups): return sorted(c for g in groups for c in GROUPS[g])
def to_dev(a, dt=torch.float32): return torch.as_tensor(a, dtype=dt, device=DEV)
def srng(*keys):
    return np.random.default_rng([int(k) if isinstance(k, (int, np.integer)) else zlib.crc32(str(k).encode()) for k in keys])

SUBSET_KEYS = [gkey(s) for s in CFG['SUBSETS']]
json.dump(CFG, open(OUT / 'logs' / 'config.json', 'w'), indent=1, default=str)
print('folds:', FOLDS, '| subsets:', SUBSET_KEYS)

# %% [markdown]
# ## 3. Loading and filtering (same filters as the paper)

# %%
def load_raw(sid):
    with open(f"{CFG['DATA_ROOT']}/S{sid}/S{sid}.pkl", 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    c = d['signal']['chest']
    x = np.concatenate([c['ACC'], c['ECG'], c['EDA'], c['EMG'], c['Resp'], c['Temp']], axis=1).astype(np.float64)
    return x, np.asarray(d['label']).astype(int).ravel()

def _filt(x, kind, f, fs=700, order=4):
    return sps.sosfiltfilt(sps.butter(order, f, btype=kind, fs=fs, output='sos'), x, axis=0)

def get_subject(sid):
    fn = OUT / 'cache' / f'S{sid}_v{CFG["DATA_VERSION"]}.npz'
    if fn.exists():
        z = np.load(fn); return z['x'], z['y']
    x, lab = load_raw(sid)
    y = np.empty_like(x)
    y[:, 0:3] = _filt(x[:, 0:3], 'low', 20); y[:, 3] = _filt(x[:, 3], 'band', [0.5, 40]); y[:, 4] = _filt(x[:, 4], 'low', 5)
    y[:, 5] = _filt(np.abs(_filt(x[:, 5], 'band', [20, 250])), 'low', 5)
    y[:, 6] = _filt(x[:, 6], 'low', 1); y[:, 7] = _filt(x[:, 7], 'low', 1)
    xp = sps.resample_poly(y, 1, 10, axis=0).astype(np.float32)
    lb = lab[::10][:len(xp)]; xp = xp[:len(lb)]
    ym = np.full(len(lb), -1, int); ym[lb == 1] = 0; ym[lb == 2] = 1; ym[lb == 3] = 2
    np.savez_compressed(fn, x=xp, y=ym)
    return xp, ym

def robust_norm(x, ref_mask):
    """label-free robust z-score: median / (IQR/1.349) of the reference samples"""
    r = x[ref_mask]
    med = np.median(r, 0); iqr = np.subtract(*np.percentile(r, [75, 25], 0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    return np.clip((x - med) / iqr, -CFG['CLIP_Z'], CFG['CLIP_Z']).astype(np.float32)

# %% [markdown]
# ## 4. Physiological feature branch (label-free, trailing context)
# The derived series are computed from the normalised 70-Hz signal of one subject:
# * **ECG**: R-peaks (sub-sample refined) → instantaneous HR, IBI, successive differences.
# * **EDA**: tonic level (0.05-Hz low-pass), phasic component and SCR events.
# * **Resp**: breaths → instantaneous respiration rate.
# * **EMG** envelope, **Temp**, and **ACC** magnitude.
#
# For each window, the features are trailing statistics over 20 s and 60 s ending at the window's last sample.
# They therefore use only the present and the past of the signal. The features are robust-normalised per subject
# with the same label-free reference as the raw signal.

# %%
FEAT_DEF = [('ECG', 'hr_mean'), ('ECG', 'hr_std'), ('ECG', 'rmssd'), ('ECG', 'ibi_std'),
            ('EDA', 'scl_mean'), ('EDA', 'scl_slope'), ('EDA', 'phasic_std'), ('EDA', 'scr_rate'),
            ('Resp', 'rr_mean'), ('Resp', 'rr_cv'), ('Resp', 'resp_std'),
            ('EMG', 'emg_mean'), ('EMG', 'emg_std'), ('Temp', 'temp_mean'), ('Temp', 'temp_slope'),
            ('ACC', 'acc_mean'), ('ACC', 'acc_std')]
FEAT_NAMES = [f'{n}_{L}s' for L in CFG['FEAT_CONTEXT_S'] for _, n in FEAT_DEF]
FEAT_GROUP = [g for L in CFG['FEAT_CONTEXT_S'] for g, _ in FEAT_DEF]

def _event_series(T, t_ev, val):
    """value defined at event times -> per-sample series (linear interpolation, constant ends)"""
    if len(t_ev) < 2: return np.zeros(T, np.float64)
    return np.interp(np.arange(T), t_ev, val)

def _roll(s, L):
    cs = np.concatenate([[0.0], np.cumsum(s)]); idx = np.arange(len(s)); lo = np.maximum(0, idx + 1 - L)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)

def _roll_std(s, L):
    return np.sqrt(np.maximum(_roll(s * s, L) - _roll(s, L) ** 2, 0))

def _roll_slope(s, L):
    h = max(1, L // 2); cs = np.concatenate([[0.0], np.cumsum(s)]); idx = np.arange(len(s))
    a = np.maximum(0, idx + 1 - h); b = np.maximum(0, idx + 1 - L)
    late = (cs[idx + 1] - cs[a]) / (idx + 1 - a)
    early = (cs[a] - cs[b]) / np.maximum(1, a - b)
    return np.where(a > b, late - early, 0.0)

def derived_series(x):
    fs = CFG['FS']; T = len(x); D = {}
    e = x[:, 3].astype(np.float64)
    if skew(e) < 0: e = -e                                         # R-peaks positive
    pk, _ = sps.find_peaks(e, distance=int(0.33 * fs), prominence=max(1e-3, 0.8 * np.std(e)))
    if len(pk) > 3:
        y0, y1, y2 = e[np.clip(pk - 1, 0, T - 1)], e[pk], e[np.clip(pk + 1, 0, T - 1)]
        den = (y0 - 2 * y1 + y2); off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        tp = (pk + np.clip(off, -0.5, 0.5)) / fs
        ibi = np.diff(tp); ok = (ibi > 0.3) & (ibi < 2.0)
        tm = tp[1:][ok] * fs; ibi = ibi[ok]
        D['hr'] = _event_series(T, tm, 60.0 / ibi) if len(ibi) > 1 else np.zeros(T)
        D['ibi'] = _event_series(T, tm, ibi) if len(ibi) > 1 else np.zeros(T)
        dd = np.diff(ibi) ** 2 if len(ibi) > 2 else np.zeros(0)
        D['sd2'] = _event_series(T, tm[1:], dd) if len(dd) > 1 else np.zeros(T)
    else:
        D['hr'] = D['ibi'] = D['sd2'] = np.zeros(T)
    eda = x[:, 4].astype(np.float64)
    tonic = sps.sosfiltfilt(sps.butter(2, 0.05, 'low', fs=fs, output='sos'), eda)
    D['tonic'] = tonic; D['phasic'] = eda - tonic
    ev = np.zeros(T); scr, _ = sps.find_peaks(D['phasic'], distance=fs, prominence=0.05); ev[scr] = 1.0
    D['scr'] = ev
    r = x[:, 6].astype(np.float64)
    br, _ = sps.find_peaks(r, distance=int(1.5 * fs), prominence=0.3)
    if len(br) > 3:
        bi = np.diff(br) / fs; ok = (bi > 1.0) & (bi < 12.0)
        D['rr'] = _event_series(T, br[1:][ok], 60.0 / bi[ok]) if ok.sum() > 1 else np.zeros(T)
    else:
        D['rr'] = np.zeros(T)
    D['resp'] = r; D['emg'] = x[:, 5].astype(np.float64); D['temp'] = x[:, 7].astype(np.float64)
    D['acc'] = np.sqrt((x[:, 0:3].astype(np.float64) ** 2).sum(1))
    return D

def feature_matrix(x):
    """per-sample feature series (T, n_feat); computed once per signal version"""
    D = derived_series(x); cols = []
    for Ls in CFG['FEAT_CONTEXT_S']:
        L = int(Ls * CFG['FS'])
        rr_m = _roll(D['rr'], L)
        cols += [_roll(D['hr'], L), _roll_std(D['hr'], L), np.sqrt(np.maximum(_roll(D['sd2'], L), 0)), _roll_std(D['ibi'], L),
                 _roll(D['tonic'], L), _roll_slope(D['tonic'], L), _roll_std(D['phasic'], L), _roll(D['scr'], L) * CFG['FS'] * 60,
                 rr_m, _roll_std(D['rr'], L) / np.maximum(np.abs(rr_m), 1e-3), _roll_std(D['resp'], L),
                 _roll(D['emg'], L), _roll_std(D['emg'], L), _roll(D['temp'], L), _roll_slope(D['temp'], L),
                 _roll(D['acc'], L), _roll_std(D['acc'], L)]
    Fm = np.stack(cols, 1)
    return np.nan_to_num(Fm, nan=0.0, posinf=0.0, neginf=0.0)

# %% [markdown]
# ## 5. Views: block-wise split, windowing, normalisation
# * **Training client:** each condition segment is split in time into its first 90% (train) and last 10% (validation),
#   before windowing. Raw signals and features are normalised with the client's whole unlabelled recording, excluding
#   its validation block.
# * **Held-out subject:** normalised with its own whole unlabelled recording (retrospective, subject-calibrated
#   normalisation, declared in the paper). Windows use one uniform stride (label-independent).
# * Signal-level corruption (scenarios) is applied **after normalisation and before feature extraction**, so the
#   feature branch sees the same corrupted signal as the raw branch.

# %%
def runs_of(y):
    out, i, n = [], 0, len(y)
    while i < n:
        if y[i] < 0: i += 1; continue
        j = i
        while j < n and y[j] == y[i]: j += 1
        out.append((i, j, int(y[i]))); i = j
    return out

def make_windows(code, stride_of_class):
    """code per sample: 0..2 class in partition, 3 unlabelled (allowed impurity), 4 forbidden.
    stride_of_class: dict class -> stride (multiples of the smallest stride)"""
    W = CFG['WIN']; n = len(code)
    base = min(stride_of_class.values())
    if n < W: return np.zeros(0, int), np.zeros(0, int)
    cs = np.stack([np.concatenate([[0], np.cumsum(code == v)]) for v in range(5)])
    starts = np.arange(0, n - W + 1, base)
    cnt = cs[:, starts + W] - cs[:, starts]
    maj = cnt[:3].argmax(0); ok = (cnt[4] == 0) & (cnt[:3].max(0) / W >= CFG['PURITY'])
    strd = np.array([stride_of_class[k] for k in range(K)])[maj]
    ok &= (starts % strd) == 0
    return starts[ok], maj[ok]

def gather(x, st):
    return x[st[:, None] + np.arange(CFG['WIN'])[None]]

def corrupt_signal(xn, spec, rng):
    """scenario corruption on the normalised signal of a corrupted client: noise on one group, optionally zero another"""
    x = xn.copy(); order = rng.permutation(len(GNAMES))
    g1 = GROUPS[GNAMES[order[0]]]; x[:, g1] += spec['sensor_noise'] * rng.standard_normal((len(x), len(g1))).astype(np.float32)
    if spec['zero_group']: x[:, GROUPS[GNAMES[order[1]]]] = 0
    return x

def build_client(sid, spec=None, rng=None):
    x, y = get_subject(sid)
    part = np.full(len(y), -1)
    for s, e, c in runs_of(y):
        cut = s + int(round((1 - CFG['VAL_FRAC']) * (e - s))); part[s:cut] = 0; part[cut:e] = 1
    ref = part != 1
    xn = robust_norm(x, ref)
    if spec is not None: xn = corrupt_signal(xn, spec, rng)
    Fm = feature_matrix(xn); Fn = robust_norm(Fm, ref)
    out = {}
    for name, p, strides in [('tr', 0, CFG['STRIDE_TRAIN']), ('va', 1, {k: CFG['STRIDE_EVAL'] for k in LAB})]:
        code = np.where(part == p, y, np.where(part == -1, 3, 4)); code[(part == p) & (y < 0)] = 3
        st, lab = make_windows(code, strides)
        out['X' + name] = gather(xn, st); out['F' + name] = Fn[st + CFG['WIN'] - 1]; out['y' + name] = lab
    return out

def build_test(sid, spec=None, rng=None):
    x, y = get_subject(sid)
    xn = robust_norm(x, np.ones(len(y), bool))
    if spec is not None and spec['test_noise'] > 0:
        xn = xn + spec['test_noise'] * rng.standard_normal(xn.shape).astype(np.float32)
    Fm = feature_matrix(xn); Fn = robust_norm(Fm, np.ones(len(y), bool))
    st, lab = make_windows(np.where(y >= 0, y, 3), {k: CFG['STRIDE_EVAL'] for k in LAB})
    X = gather(xn, st); Fw = Fn[st + CFG['WIN'] - 1].copy()
    if spec is not None and spec['test_group_drop'] > 0:              # missing sensor group in a window
        for g in GNAMES:
            drop = rng.random(len(X)) < spec['test_group_drop']
            X[np.ix_(drop, np.arange(X.shape[1]), GROUPS[g])] = 0
            Fw[np.ix_(drop, [i for i, fg in enumerate(FEAT_GROUP) if fg == g])] = 0
    return dict(X=X, F=Fw, y=lab, start=st, t=st / CFG['FS'], ylab=y, xn=xn, Fn_s=Fn[CFG['WIN'] - 1::CFG['FS']].copy())

def calib_view(sid, cal_s):
    """subject-calibrated protocol: labelled first cal_s seconds of each condition (train), buffer, later windows (test)"""
    T = TEST[sid]; y = T['ylab']; fs = CFG['FS']; W = CFG['WIN']
    cal_code = np.full(len(y), 4); ev_ok = np.zeros(len(y), bool)
    for s, e, c in runs_of(y):
        a = min(e, s + int(cal_s * fs)); b = min(e, a + int(CFG['CAL_BUFFER_S'] * fs))
        cal_code[s:a] = c; ev_ok[b:e] = True
    st, lab = make_windows(cal_code, {k: fs for k in LAB})                 # 1-s stride inside the calibration minutes
    keep = np.array([ev_ok[s0:s0 + W].all() for s0 in T['start']], bool)   # test windows entirely after the buffer
    return dict(Xc=gather(T['xn'], st), Fc=T['Fn_s'][st // fs], yc=lab, keep=keep)   # starts are multiples of fs

t0 = time.time()
CLIENT = {s: build_client(s) for s in CFG['SUBJECTS']}
TEST = {s: build_test(s) for s in CFG['SUBJECTS']}
print(f'views built in {time.time() - t0:.0f}s | features: {len(FEAT_NAMES)}')
rows = []
for s in CFG['SUBJECTS']:
    r = {'subject': f'S{s}'}
    for nm, yy in [('train', CLIENT[s]['ytr']), ('val', CLIENT[s]['yva']), ('test', TEST[s]['y'])]:
        for k, c in enumerate(CLASSES): r[f'{nm}_{c}'] = int((yy == k).sum())
    rows.append(r)
WC = pd.DataFrame(rows); WC.loc[len(WC)] = ['Total'] + WC.iloc[:, 1:].sum().tolist()
WC.to_csv(OUT / 'tables' / 'window_counts.csv', index=False); print(WC.tail(1).to_string(index=False))

# %% [markdown]
# ### 5b. Clock-feature rule (label-free, per fold, training subjects only)
# For each fold, the Spearman correlation between every feature and the window time is computed within each
# **training** subject (the held-out subject is excluded). Features whose mean |ρ| is ≥ `CLOCK_RHO` are dropped
# for that fold. No class label is used. The rule removes features that mainly encode *position in the session*.
# WESAD's condition order differs between subjects (B→S→A versus B→A→S), so such features generalise badly.

# %%
from scipy.stats import spearmanr
_RHO = {s: np.array([abs(spearmanr(TEST[s]['F'][:, j], TEST[s]['t']).correlation) if np.std(TEST[s]['F'][:, j]) > 0 else 0.0
                     for j in range(len(FEAT_NAMES))]) for s in CFG['SUBJECTS']}
def clock_drop(test_sid):
    r = np.nanmean([_RHO[s] for s in CFG['SUBJECTS'] if s != test_sid], 0)
    return [j for j in range(len(FEAT_NAMES)) if r[j] >= CFG['CLOCK_RHO']]
_dropcount = pd.Series([FEAT_NAMES[j] for s in FOLDS for j in clock_drop(s)]).value_counts()
print('features dropped by the clock rule (number of folds):'); print(_dropcount.to_string())
_dropcount.to_csv(OUT / 'tables' / 'v2_clock_dropped_features.csv')

def feat_idx(groups, test_sid):
    drop = set(clock_drop(test_sid))
    fidx = [i for i, g in enumerate(FEAT_GROUP) if g in groups and i not in drop]
    if not fidx:                                                   # never leave the feature branch empty
        fidx = [min((i for i, g in enumerate(FEAT_GROUP) if g in groups), key=lambda i: _RHO[test_sid][i])]
    return fidx

# %% [markdown]
# ## 6. TRACE-H: TRACE raw branch + physiological feature branch
# The raw branch is the TRACE encoder (1×1 projection to 32, five dilated causal TCN blocks, BiGRU 64×2,
# SE, temporal pooling). BatchNorm is replaced by GroupNorm (8 groups), so no batch statistics are involved. The feature branch is BN → 64 → 64. A fusion MLP produces the three classes.
# Modality dropout (raw 10%, features 15%) prevents either branch from dominating.

# %%
class TCNBlock(nn.Module):
    def __init__(self, c, d, k=3, p=0.1):
        super().__init__(); self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c, c, k, dilation=d); self.b1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, k, dilation=d); self.b2 = nn.GroupNorm(8, c); self.drop = nn.Dropout(p)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.drop(F.relu(self.b2(self.c2(F.pad(self.drop(h), (self.pad, 0))))))
        return F.relu(x + h)

class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__(); self.f = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x): return x * self.f(x.mean(-1)).unsqueeze(-1)

class TRACEH(nn.Module):
    def __init__(self, c_in, f_in, d=32, h=64):
        super().__init__()
        self.proj = nn.Conv1d(c_in, d, 1)
        self.tcn = nn.Sequential(*[TCNBlock(d, 2 ** l) for l in range(5)])
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.se = SE(2 * h)
        self.fbr = nn.Sequential(nn.LayerNorm(f_in), nn.Linear(f_in, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * h + 64, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, K))
    def forward(self, x, f):
        r = self.tcn(self.proj(x.transpose(1, 2)))
        r = self.se(self.gru(r.transpose(1, 2))[0].transpose(1, 2)).mean(-1)
        z = self.fbr(f)
        if self.training:
            B = x.shape[0]; u = torch.rand(B, device=x.device)
            dr = (u < CFG['MOD_DROP'][0]).float()[:, None]
            dz = ((u >= CFG['MOD_DROP'][0]) & (u < sum(CFG['MOD_DROP']))).float()[:, None]
            r = r * (1 - dr); z = z * (1 - dz)
        return self.head(torch.cat([r, z], 1))

def n_params(m): return sum(p.numel() for p in m.parameters())
_m = TRACEH(4, 16)
print('TRACE-H parameters (4 raw channels, 16 features):', n_params(_m),
      '| raw branch:', sum(n_params(getattr(_m, k)) for k in ['proj', 'tcn', 'gru', 'se']),
      '| feature branch:', n_params(_m.fbr), '| head:', n_params(_m.head))

def is_bn_buffer(k): return k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked')

# %% [markdown]
# ## 7. Minority-class augmentation and class-balanced sampling (inside each client)
# * **Class-balanced sampling:** each local epoch draws n windows with equal probability per class, so Amusement and
#   Stress windows are revisited more often.
# * **Class-dependent augmentation probability:** p_k = min(0.95, 0.5·√(n_max/n_k)), which is higher for rarer classes,
#   so repeated minority windows are almost always seen in a new form.
# * Augmentations (partner = same class, same client, never another subject):
#   - **TIA**, a mix of raw and feature vectors with the same λ ∈ [0.6, 1];
#   - **PCR**, which swaps whole channel groups, raw and their features together;
#   - per-channel **magnitude scaling** (±10%) and **jitter**, plus **feature jitter**;
#   - **channel masking**, applied to every window with p = 0.1.

# %%
class ClientData:
    def __init__(self, sid, ch, fidx, view):
        self.sid = sid
        self.Xtr = to_dev(view['Xtr'][:, :, ch]); self.Ftr = to_dev(view['Ftr'][:, fidx]); self.ytr = to_dev(view['ytr'], torch.long)
        self.Xva = to_dev(view['Xva'][:, :, ch]); self.Fva = to_dev(view['Fva'][:, fidx]); self.yva = np.asarray(view['yva'])
        self.refresh()
    def refresh(self):
        self.n = len(self.ytr)
        cnt = torch.bincount(self.ytr, minlength=K).float()
        self.pool_idx = [torch.nonzero(self.ytr == k).ravel() for k in range(K)]
        self.pool_idx = [i if len(i) else torch.arange(self.n, device=DEV) for i in self.pool_idx]
        w = torch.where(cnt > 0, 1.0 / cnt.clamp(min=1), torch.zeros_like(cnt))
        self.samp_w = w[self.ytr]
        A = CFG['AUG']; nmax = cnt.max()
        self.p_aug = torch.clamp(A['base_p'] * torch.sqrt(nmax / cnt.clamp(min=1)), max=A['max_p'])

def group_positions(ch, fidx):
    out = []
    for g in GNAMES:
        rp = [ch.index(c) for c in GROUPS[g] if c in ch]
        fp = [fidx.index(i) for i, fg in enumerate(FEAT_GROUP) if fg == g and i in fidx]
        if rp: out.append((rp, fp))
    return out

def augment(cd, xb, fb, yb, gpos):
    A = CFG['AUG']; B = len(yb); dv = xb.device
    partner = torch.empty(B, dtype=torch.long, device=dv)
    for k in range(K):
        m = yb == k
        if m.any(): idx = cd.pool_idx[k]; partner[m] = idx[torch.randint(len(idx), (int(m.sum()),), device=dv)]
    xp, fp = cd.Xtr[partner], cd.Ftr[partner]
    doaug = torch.rand(B, device=dv) < cd.p_aug[yb]
    x, f = xb.clone(), fb.clone()
    mix = doaug & (torch.rand(B, device=dv) < A['p_mix'])
    if mix.any():
        lam = torch.empty(B, device=dv).uniform_(A['lam_lo'], A['lam_hi'])
        x = torch.where(mix[:, None, None], lam[:, None, None] * x + (1 - lam[:, None, None]) * xp, x)
        f = torch.where(mix[:, None], lam[:, None] * f + (1 - lam[:, None]) * fp, f)
    pcr = doaug & (torch.rand(B, device=dv) < A['p_pcr'])
    if pcr.any() and len(gpos) > 1:
        for rp, fpos in gpos:                                  # swap whole channel groups (raw + their features)
            idx = (pcr & (torch.rand(B, device=dv) < A['p_group'])).nonzero().ravel()
            if len(idx):
                xs = x[idx]; xs[:, :, rp] = xp[idx][:, :, rp]; x[idx] = xs
                if fpos:
                    fs_ = f[idx]; fs_[:, fpos] = fp[idx][:, fpos]; f[idx] = fs_
    sc = 1 + A['scale'] * (2 * torch.rand(B, 1, x.shape[2], device=dv) - 1)
    x = torch.where(doaug[:, None, None], x * sc + A['jitter'] * torch.randn_like(x), x)
    f = torch.where(doaug[:, None], f + A['feat_jitter'] * torch.randn_like(f), f)
    keep = (torch.rand(B, 1, x.shape[2], device=dv) >= A['p_mask']).float()
    return x * keep, f

# %% [markdown]
# ## 8. Metrics (identical definitions for every table)

# %%
def metrics(y, yhat, P=None):
    y = np.asarray(y); yhat = np.asarray(yhat)
    cm = confusion_matrix(y, yhat, labels=LAB); n = cm.sum()
    p, r, f, s = precision_recall_fscore_support(y, yhat, labels=LAB, zero_division=0)
    pres = s > 0
    spec = np.array([(n - cm[k].sum() - cm[:, k].sum() + cm[k, k]) / max(1, n - cm[k].sum()) for k in LAB])
    out = dict(n=int(n), acc=np.trace(cm) / max(1, n), prec=p[pres].mean(), rec=r[pres].mean(), f1=f[pres].mean(),
               bal_acc=r[pres].mean(), spec=spec[pres].mean(),
               kappa=cohen_kappa_score(y, yhat, labels=LAB), mcc=matthews_corrcoef(y, yhat),
               auc_m=np.nan, auc_w=np.nan, cm=cm, prec_k=p, rec_k=r, f1_k=f, sup_k=s)
    out['gap'] = abs(out['acc'] - out['bal_acc'])
    if P is not None and pres.all():
        out['auc_m'] = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=LAB)
        out['auc_w'] = roc_auc_score(y, P, multi_class='ovr', average='weighted', labels=LAB)
    return out

def macro_f1_from_cm(cm):
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), 0)
    return f1[cm.sum(1) > 0].mean()

# %% [markdown]
# ## 9. Federated training with PRISM-RM
# Per round:
# 1. Each participating client loads the global weights and trains locally (balanced sampling with augmentation).
# 2. The client computes per-class F1 on its validation block, updates its EMA, and sends **one reliability bit**
#    q_i = 1[min_k F1 < τ]. No bit is sent in the 3 warm-up rounds.
# 3. The server aggregates every parameter by the **reliability-weighted coordinate-wise median**, with weights n_i·(1 − 0.7·q_i).
# 4. Checkpoint score: macro-F1 from the **sum of client validation confusion matrices**. The 3 best rounds are kept
#    for the snapshot ensemble, with early stopping after 10 rounds without improvement.
# 5. **Prior correction:** for τ_p ∈ {0, 0.5, 1}, the ensemble's validation predictions (logits + τ_p·log π) are scored
#    with Cohen's κ on the training clients' validation blocks. The best τ_p is then applied to the held-out subject.
#    This is validation tuning only.
#
# The model has no BatchNorm (GroupNorm / LayerNorm), so no statistics need to be shared or re-estimated.
# The FedBN/AdaBN code remains but is switched off.

# %%
def get_state(m): return {k: v.detach().clone() for k, v in m.state_dict().items()}

def weighted_median(S, w):
    """S (n, ...), w (n,) -> coordinate-wise weighted median"""
    n = S.shape[0]; flat = S.reshape(n, -1)
    vals, idx = flat.sort(0)
    cw = w[idx].cumsum(0)
    k = (cw < 0.5 * cw[-1:]).sum(0, keepdim=True).clamp(max=n - 1)
    return vals.gather(0, k).reshape(S.shape[1:])

def aggregate(states, w, g):
    w_t = torch.tensor(w, dtype=torch.float32, device=DEV); w_t = w_t / w_t.sum()
    out = {}
    for k in g:
        if is_bn_buffer(k):
            if k.endswith('num_batches_tracked'): out[k] = g[k]
            else: out[k] = sum(float(wi) * s[k] for wi, s in zip(w_t, states))   # only used to initialise new clients
            continue
        S = torch.stack([s[k] for s in states])
        out[k] = weighted_median(S, w_t) if CFG['AGG'] == 'prism_wmedian' else (w_t.view(-1, *[1] * (S.dim() - 1)) * S).sum(0)
    return out

def local_train(model, cd, lr, epochs, gpos):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG['WD'])
    for _ in range(epochs):
        order = torch.multinomial(cd.samp_w, cd.n, replacement=True)
        for i in range(0, cd.n, CFG['BATCH']):
            b = order[i:i + CFG['BATCH']]
            if len(b) < 2: continue
            xb, fb = augment(cd, cd.Xtr[b], cd.Ftr[b], cd.ytr[b], gpos)
            loss = F.cross_entropy(model(xb, fb), cd.ytr[b], label_smoothing=CFG['LABEL_SMOOTH'])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

EVAL_BS = [int(os.environ.get('FEDPHYSIO_EVAL_BS', '128'))]     # evaluation batch size; halves itself on CUDA OOM

@torch.no_grad()
def _forward_batches(model, X, Fe, fn):
    model.eval()
    X = X if torch.is_tensor(X) else to_dev(X); Fe = Fe if torch.is_tensor(Fe) else to_dev(Fe)
    while True:
        try:
            out = [fn(model(X[i:i + EVAL_BS[0]], Fe[i:i + EVAL_BS[0]])).cpu() for i in range(0, len(X), EVAL_BS[0])]
            return torch.cat(out).numpy() if out else np.zeros((0, K))
        except RuntimeError as e:
            if 'out of memory' not in str(e) or EVAL_BS[0] <= 8: raise
            torch.cuda.empty_cache(); EVAL_BS[0] //= 2
            print(f'      CUDA OOM during evaluation -> batch size {EVAL_BS[0]}', flush=True)

def predict(model, X, Fe):
    return _forward_batches(model, X, Fe, lambda z: F.softmax(z, -1))

def predict_logits(model, X, Fe):
    return _forward_batches(model, X, Fe, lambda z: z)

def softmax_np(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

@torch.no_grad()
def adabn(model, X, Fe, bs=512):
    """re-estimate BatchNorm statistics from unlabelled windows (cumulative average); no labels, no gradients"""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns: m.reset_running_stats(); m.momentum = None
    model.eval()
    for m in bns: m.train()
    X = X if torch.is_tensor(X) else to_dev(X); Fe = Fe if torch.is_tensor(Fe) else to_dev(Fe)
    for i in range(0, len(X), bs):
        if len(X[i:i + bs]) > 1: model(X[i:i + bs], Fe[i:i + bs])
    for m in bns: m.momentum = 0.1
    model.eval()

def _save_ckpt(path, obj):
    tmp = str(path) + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)

def _load(path):
    try: return torch.load(path, map_location=DEV, weights_only=False)
    except TypeError: return torch.load(path, map_location=DEV)

def make_clients(test_sid, ch, fidx, scenario, seed):
    spec = CFG['SCENARIOS'][scenario]
    others = [s for s in CFG['SUBJECTS'] if s != test_sid]
    rng = srng(seed, test_sid, 'corrupt', scenario)
    bad = sorted(rng.choice(len(others), min(spec['n_corrupt'], len(others) - 1), replace=False).tolist()) if spec['n_corrupt'] else []
    clients = []
    for i, s in enumerate(others):
        if i in bad:
            view = build_client(s, spec, srng(seed, test_sid, 'sig', s, scenario))
            yr = srng(seed, test_sid, 'lab', s, scenario)
            for key in ['ytr', 'yva']:
                y = view[key].copy(); flip = yr.random(len(y)) < spec['label_noise']
                y[flip] = (y[flip] + yr.integers(1, K, flip.sum())) % K; view[key] = y
        else:
            view = CLIENT[s]
        clients.append(ClientData(s, ch, fidx, view))
    return clients, [others[i] for i in bad]

def run_fl(test_sid, groups, scenario='good', seed=42, ckpt=None):
    set_seed(seed)
    ch = chans(groups); fidx = feat_idx(groups, test_sid)
    spec = CFG['SCENARIOS'][scenario]
    clients, corrupted = make_clients(test_sid, ch, fidx, scenario, seed)
    tv = TEST[test_sid] if scenario == 'good' else build_test(test_sid, spec, srng(seed, test_sid, 'test', scenario))
    tv = {k: tv[k] for k in ['X', 'F', 'y', 't']}
    rng_part = srng(seed, test_sid, 'participation', scenario)
    N = len(clients); n_part = max(2, int(round(spec['participation'] * N)))
    gpos = group_positions(ch, fidx)
    model = TRACEH(len(ch), len(fidx)).to(DEV)
    g = get_state(model)
    bn_keys = [k for k in g if is_bn_buffer(k)]
    local_bn = {}                                            # FedBN: BN statistics per client (stay on the client)
    ema = {}
    snaps, best, wait = [], -1.0, 0                          # snaps: list of (val, round, state, client_bn)
    hist = dict(val_f1=[], flags=[], screened=[], corrupted=[int(c) for c in corrupted], best_round=0)
    R = CFG['R_MAX']; start = 0
    if ckpt is not None and Path(ckpt).exists():
        ck = _load(ckpt)
        g, local_bn, ema, snaps, best, wait, hist, start = (ck[k] for k in ['g', 'local_bn', 'ema', 'snaps', 'best', 'wait', 'hist', 't'])
        start += 1; rng_part.bit_generator.state = ck['rng_part']
        torch.set_rng_state(ck['torch_rng'].cpu().to(torch.uint8)); np.random.set_state(ck['np_rng']); random.setstate(ck['py_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ck['cuda_rng']])
        if ck.get('stopped'): start = R
        print(f'      resumed after round {start}', flush=True)

    def load_for(cd):
        st = dict(g)
        if CFG['FEDBN'] and cd.sid in local_bn: st.update(local_bn[cd.sid])
        model.load_state_dict(st)

    for t in range(start, R):
        lr = CFG['LR'] * 0.5 * (1 + math.cos(math.pi * t / R))
        part = sorted(rng_part.choice(N, n_part, replace=False).tolist()) if n_part < N else list(range(N))
        states, w, flags = [], [], {}
        for ci in part:
            cd = clients[ci]; load_for(cd)
            local_train(model, cd, lr, CFG['LOCAL_EPOCHS'], gpos)
            yv = predict(model, cd.Xva, cd.Fva).argmax(1)
            f1k = precision_recall_fscore_support(cd.yva, yv, labels=LAB, zero_division=0)[2]
            ema[cd.sid] = f1k if cd.sid not in ema else CFG['EMA'] * ema[cd.sid] + (1 - CFG['EMA']) * f1k
            present = np.bincount(cd.yva, minlength=K) > 0
            q = 0 if t < CFG['WARMUP'] else int(ema[cd.sid][present].min() < CFG['TAU'])
            flags[int(cd.sid)] = q
            s_i = get_state(model)
            local_bn[cd.sid] = {k: s_i[k].clone() for k in bn_keys}
            states.append(s_i); w.append(cd.n * (1 - (1 - CFG['PENALTY']) * q))
        g = aggregate(states, w, g)
        # checkpoint score: summed validation confusion matrices (each client with its own BN statistics)
        cm = np.zeros((K, K), int)
        for cd in clients:
            load_for(cd); cm += confusion_matrix(cd.yva, predict(model, cd.Xva, cd.Fva).argmax(1), labels=LAB)
        v = float(macro_f1_from_cm(cm))
        hist['val_f1'].append(v); hist['flags'].append(flags); hist['screened'].append(int(sum(flags.values())))
        if not np.isfinite(sum(float(x.float().sum()) for k, x in g.items() if not is_bn_buffer(k))): break
        snaps.append((v, t + 1, {k: x.clone() for k, x in g.items()}))
        snaps = sorted(snaps, key=lambda s: -s[0])[:CFG['N_SNAPSHOT']]
        if v > best: best, wait, hist['best_round'] = v, 0, t + 1
        else: wait += 1
        stop = wait >= CFG['PATIENCE']
        if ckpt is not None:
            _save_ckpt(ckpt, dict(g=g, local_bn=local_bn, ema=ema, snaps=snaps, best=best, wait=wait, hist=hist, t=t, stopped=stop,
                                  rng_part=rng_part.bit_generator.state, torch_rng=torch.get_rng_state(),
                                  np_rng=np.random.get_state(), py_rng=random.getstate(),
                                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        if stop: break
    # ---- prior correction tau_p chosen on VALIDATION kappa (training clients), then applied to the test subject ----
    Xte, Fte = to_dev(tv['X'][:, :, ch]), to_dev(tv['F'][:, fidx])
    yv_all = np.concatenate([cd.yva for cd in clients])
    prior = np.bincount(yv_all, minlength=K) + 1.0; logpi = np.log(prior / prior.sum())   # = sum of client validation counts
    Lv, Lt, states_out = [], [], []
    for v, r, st in snaps:
        model.load_state_dict(st)
        Lv.append(np.concatenate([predict_logits(model, cd.Xva, cd.Fva) for cd in clients]))
        Lt.append(predict_logits(model, Xte, Fte)); states_out.append(get_state(model))
    def ens(L, tp): return np.mean([softmax_np(l + tp * logpi) for l in L], 0)
    vk = {tp: cohen_kappa_score(yv_all, ens(Lv, tp).argmax(1), labels=LAB) for tp in CFG['PRIOR_TAUS']}
    tp_best = max(CFG['PRIOR_TAUS'], key=lambda tp: (round(vk[tp], 4), -tp))
    hist['prior_val_kappa'] = {str(k): float(v) for k, v in vk.items()}
    return dict(P=ens(Lt, tp_best), P_best=softmax_np(Lt[0] + tp_best * logpi), P_tau0=ens(Lt, 0.0),
                y=tv['y'], t=tv['t'], hist=hist, best_val=best, snap_rounds=[r for _, r, _ in snaps],
                states=states_out, ch=ch, fidx=fidx, dropped=[FEAT_NAMES[j] for j in clock_drop(test_sid)],
                prior_tau=tp_best, logpi=logpi.tolist())

# %% [markdown]
# ## 10. Run plan: per fold, 8 subsets → validation choice → average and worst scenarios
# The subset of fold f is the one with the highest **internal-validation** macro-F1. Ties go to fewer channels.
# The test subject plays no part in this choice. The other subsets' held-out results are reported afterwards
# as a sensitivity table only.

# %%
def run_dir(name, seed): d = OUT / 'runs' / f'{name}_r{CFG["RUN_VERSION"]}_seed{seed}'; d.mkdir(parents=True, exist_ok=True); return d
def done(name, sid, seed): return (run_dir(name, seed) / f'S{sid}.csv').exists()
def hist_of(name, sid, seed): return json.load(open(run_dir(name, seed) / f'S{sid}_hist.json'))

def save_run(name, sid, seed, res, extra):
    d = run_dir(name, seed); P = res['P']
    pd.DataFrame(dict(y_true=res['y'], y_pred=P.argmax(1), p0=P[:, 0], p1=P[:, 1], p2=P[:, 2],
                      y_pred_best=res['P_best'].argmax(1), y_pred_tau0=res['P_tau0'].argmax(1), t_start=res['t'])).to_csv(d / f'S{sid}.csv', index=False)
    json.dump(dict(hist=res['hist'], best_val=res['best_val'], snap_rounds=res['snap_rounds'], logpi=res['logpi'], extra=extra),
              open(d / f'S{sid}_hist.json', 'w'), default=float)

class Timer:
    def __init__(self, total, already): self.total, self.n, self.t = total, already, []
    def tick(self, dt):
        self.n += 1; self.t.append(dt); eta = np.mean(self.t[-20:]) * max(0, self.total - self.n) / 3600
        return f'[{self.n}/{self.total}, ETA {eta:.1f} h]'

WID, NW = map(int, CFG['WORKER'].split('/'))
MY_FOLDS = [f for i, f in enumerate(FOLDS) if i % NW == WID]
SCN_OTHER = [s for s in CFG['SCENARIOS'] if s != 'good']
def cfg_name(subset_key, scenario='good'): return f'sub_{subset_key}' + ('' if scenario == 'good' else f'__{scenario}')
_total = len(MY_FOLDS) * len(CFG['SEEDS']) * (len(SUBSET_KEYS) + len(SCN_OTHER))
_done = sum(done(cfg_name(k), s, sd) for sd in CFG['SEEDS'] for s in MY_FOLDS for k in SUBSET_KEYS)
TIMER = Timer(_total, _done)
print(f'worker {WID}/{NW}: folds {MY_FOLDS} | planned runs {_total} | already done (subsets) {_done}')

def execute(name, sid, seed, groups, scenario='good'):
    if done(name, sid, seed): return
    t0 = time.time()
    sig = hashlib.md5(json.dumps([name, groups, scenario, CFG['DATA_VERSION'], CFG['RUN_VERSION'], CFG['AGG']]).encode()).hexdigest()[:8]
    ck = OUT / 'ckpt' / f'{name}_S{sid}_seed{seed}_{sig}.pt'
    res = run_fl(sid, groups, scenario, seed, ckpt=ck)
    torch.save(res['states'], OUT / 'states' / f'{name}_r{CFG["RUN_VERSION"]}_S{sid}_seed{seed}.pt')
    save_run(name, sid, seed, res, dict(groups=groups, scenario=scenario, data_version=CFG['DATA_VERSION'], run_version=CFG['RUN_VERSION'], agg=CFG['AGG'],
                                          dropped_features=res['dropped'], prior_tau=res['prior_tau']))
    ck.unlink(missing_ok=True)
    del res['states']
    if DEV.type == 'cuda': torch.cuda.empty_cache()
    acc = (res['P'].argmax(1) == res['y']).mean()
    print(f'  {name:40s} S{sid:<3d} acc={acc:.4f} val={res["best_val"]:.4f} tau_p={res["prior_tau"]} snaps={res["snap_rounds"]} '
          f'{time.time()-t0:5.0f}s {TIMER.tick(time.time()-t0)}', flush=True)

def select_subset(sid, seed):
    """internal validation only; ties -> fewer channels"""
    val = {k: hist_of(cfg_name(k), sid, seed)['best_val'] for k in SUBSET_KEYS}
    nch = {k: len(chans(g)) for k, g in zip(SUBSET_KEYS, CFG['SUBSETS'])}
    return max(SUBSET_KEYS, key=lambda k: (round(val[k], 4), -nch[k])), val

for seed in CFG['SEEDS']:
    for sid in MY_FOLDS:
        print(f'=== fold S{sid} (seed {seed}) ===', flush=True)
        for key, groups in zip(SUBSET_KEYS, CFG['SUBSETS']):
            execute(cfg_name(key), sid, seed, groups)
        sel, val = select_subset(sid, seed)
        json.dump(dict(selected=sel, val=val), open(run_dir('selection', seed) / f'S{sid}.json', 'w'), indent=1)
        print(f'  selected on validation: {sel}', flush=True)
        groups = CFG['SUBSETS'][SUBSET_KEYS.index(sel)]
        for sc in SCN_OTHER:
            execute(cfg_name(sel, sc), sid, seed, groups, sc)
print('this worker has finished its federated runs')

# %% [markdown]
# ## 11. Wait for all workers, then assemble FedPhysio v2 (strict LOSO)

# %%
S0 = CFG['SEEDS'][0]
def wait_all():
    t0 = time.time()
    while True:
        miss = [s for s in FOLDS if not (run_dir('selection', S0) / f'S{s}.json').exists()]
        if not miss: return
        if time.time() - t0 < 2: print('waiting for other workers:', miss, flush=True)
        time.sleep(60)
wait_all()
SEL = {s: json.load(open(run_dir('selection', S0) / f'S{s}.json'))['selected'] for s in FOLDS}

def load_preds(name, sid, seed=S0): return pd.read_csv(run_dir(name, seed) / f'S{sid}.csv')

def smooth_causal(P, t):
    """trailing average of probabilities over the past SMOOTH_S seconds (past outputs only)"""
    order = np.argsort(t); Ps = np.empty_like(P); tt = t[order]; PP = P[order]
    cs = np.vstack([np.zeros((1, K)), np.cumsum(PP, 0)])
    lo = np.searchsorted(tt, tt - CFG['SMOOTH_S'] + 1e-9, side='left')
    Ps[order] = (cs[np.arange(len(tt)) + 1] - cs[lo]) / (np.arange(len(tt)) + 1 - lo)[:, None]
    return Ps

def collect(get):
    """get(sid) -> (y, P); returns per-subject metrics, pooled metrics"""
    subj, Y, PP = {}, [], []
    for s in FOLDS:
        y, P = get(s); subj[s] = metrics(y, P.argmax(1), P); Y.append(y); PP.append(P)
    Y = np.concatenate(Y); PP = np.concatenate(PP)
    return subj, metrics(Y, PP.argmax(1), PP)

def strict(s):
    d = load_preds(cfg_name(SEL[s]), s); return d.y_true.values, d[['p0', 'p1', 'p2']].values
def strict_smooth(s):
    d = load_preds(cfg_name(SEL[s]), s); return d.y_true.values, smooth_causal(d[['p0', 'p1', 'p2']].values, d.t_start.values)

RES = {}
RES['strict'] = collect(strict)
RES['strict_smooth'] = collect(strict_smooth)

def check(subj, pooled):
    s, r = pooled['sup_k'], pooled['rec_k']
    ok = abs((s * r).sum() / s.sum() - pooled['acc']) < 1e-12
    ok &= (sum(m['cm'] for m in subj.values()) == pooled['cm']).all()
    return bool(ok)
print('consistency checks:', {k: check(*v) for k, v in RES.items()})
for k, (subj, pl) in RES.items():
    print(f'{k:15s} acc={pl["acc"]:.4f} F1={pl["f1"]:.4f} kappa={pl["kappa"]:.4f} AUC={pl["auc_m"]:.4f} '
          f'subject-kappa {np.mean([m["kappa"] for m in subj.values()]):.4f}±{np.std([m["kappa"] for m in subj.values()], ddof=1):.4f}')
print('selected subsets:', SEL)

# %% [markdown]
# ## 12. Subject-calibrated protocol (separate table)
# For each held-out subject and each N in `CAL_S`:
# * The **calibration set** is the first N seconds of each condition (labelled), windowed with a 1-s stride.
# * The **buffer** is the next 60 s, which are never tested.
# * The **evaluation set** is all later test windows (same uniform stride as strict LOSO).
#
# Each snapshot of the selected model is fine-tuned on the calibration set: feature branch,
# fusion head and GroupNorm affine parameters; the rest of the raw encoder is frozen. The same prior correction τ_p is applied. The fine-tuned snapshots are then averaged.
# Strict LOSO on **the same evaluation windows** is reported alongside, so the calibration gain is a like-for-like comparison.

# %%
def calibrate(s, cal_s, seed=S0):
    key = SEL[s]; groups = CFG['SUBSETS'][SUBSET_KEYS.index(key)]
    ch = chans(groups); fidx = feat_idx(groups, s)
    cv = calib_view(s, cal_s); T = TEST[s]
    h = hist_of(cfg_name(key), s, seed); tp = h['extra']['prior_tau']; logpi = np.array(h['logpi'])
    states = _load(OUT / 'states' / f'{cfg_name(key)}_r{CFG["RUN_VERSION"]}_S{s}_seed{seed}.pt')
    Xc, Fc, yc = to_dev(cv['Xc'][:, :, ch]), to_dev(cv['Fc'][:, fidx]), to_dev(cv['yc'], torch.long)
    Xe, Fe = to_dev(T['X'][cv['keep']][:, :, ch]), to_dev(T['F'][cv['keep']][:, fidx])
    cd = type('C', (), {})(); cd.Xtr, cd.Ftr, cd.ytr = Xc, Fc, yc; ClientData.refresh(cd)
    gpos = group_positions(ch, fidx); Ps = []
    for st in states:
        set_seed(seed); m = TRACEH(len(ch), len(fidx)).to(DEV); m.load_state_dict(st)
        train_p = [p for n, p in m.named_parameters() if n.startswith(('fbr', 'head')) or re.match(r'tcn\.\d+\.b[12]\.', n)]
        for p in m.parameters(): p.requires_grad_(False)
        for p in train_p: p.requires_grad_(True)
        opt = torch.optim.AdamW(train_p, lr=CFG['CAL_LR'], weight_decay=CFG['WD'])
        for _ in range(CFG['CAL_EPOCHS']):
            m.train()
            for mod in m.modules():                                 # keep the adapted BN statistics fixed
                if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.eval()
            order = torch.multinomial(cd.samp_w, cd.n, replacement=True)
            for i in range(0, cd.n, 32):
                b = order[i:i + 32]
                if len(b) < 2: continue
                xb, fb = augment(cd, Xc[b], Fc[b], yc[b], gpos)
                loss = F.cross_entropy(m(xb, fb), yc[b], label_smoothing=CFG['LABEL_SMOOTH'])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        Ps.append(softmax_np(predict_logits(m, Xe, Fe) + tp * logpi))
    d = load_preds(cfg_name(key), s)
    P_strict = d[['p0', 'p1', 'p2']].values[cv['keep']]
    return T['y'][cv['keep']], np.mean(Ps, 0), P_strict, int(len(cv['yc']))

CAL = {}
for cs in CFG['CAL_S']:
    fn = OUT / 'runs' / f'calibrated_{cs}s_r{CFG["RUN_VERSION"]}_seed{S0}.pkl'
    if fn.exists(): CAL[cs] = pickle.load(open(fn, 'rb')); continue
    out = {}
    for s in FOLDS:
        out[s] = calibrate(s, cs); print(f'  calibrated {cs}s S{s}: acc={(out[s][1].argmax(1) == out[s][0]).mean():.4f} '
                                         f'(strict on same windows {(out[s][2].argmax(1) == out[s][0]).mean():.4f}, n_cal={out[s][3]})', flush=True)
    pickle.dump(out, open(fn, 'wb')); CAL[cs] = out
for cs in CFG['CAL_S']:
    RES[f'calibrated_{cs}s'] = collect(lambda s: (CAL[cs][s][0], CAL[cs][s][1]))
    RES[f'strict_same_windows_{cs}s'] = collect(lambda s: (CAL[cs][s][0], CAL[cs][s][2]))

# %% [markdown]
# ## 13. Tables (LaTeX bodies and CSV in `fedphysio_v2_results/tables/`)

# %%
def f4(x): return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'
def write(name, lines): (OUT / 'tables' / name).write_text('\n'.join(lines) + '\n'); print(f'--- {name}'); print('\n'.join(lines))
PROT_LABEL = {'strict': 'Strict LOSO (window-level)', 'strict_smooth': f'Strict LOSO + causal {CFG["SMOOTH_S"]}-s smoothing'}
for cs in CFG['CAL_S']:
    PROT_LABEL[f'strict_same_windows_{cs}s'] = f'Strict LOSO, same windows as calibrated-{cs}s'
    PROT_LABEL[f'calibrated_{cs}s'] = f'Subject-calibrated ({cs} s per condition)'

# T1: all protocols, pooled + subject summary
rows, lines = [], []
for k, lab in PROT_LABEL.items():
    subj, pl = RES[k]; kk = np.array([m['kappa'] for m in subj.values()]); aa = np.array([m['acc'] for m in subj.values()])
    rows.append(dict(protocol=lab, n=pl['n'], acc=pl['acc'], bal_acc=pl['bal_acc'], prec=pl['prec'], rec=pl['rec'], f1=pl['f1'],
                     spec=pl['spec'], auc_m=pl['auc_m'], auc_w=pl['auc_w'], kappa=pl['kappa'], mcc=pl['mcc'], gap=pl['gap'],
                     subj_acc_mean=aa.mean(), subj_acc_sd=aa.std(ddof=1), subj_kappa_mean=kk.mean(), subj_kappa_sd=kk.std(ddof=1),
                     worst_kappa=kk.min(), worst_subject=f'S{FOLDS[int(kk.argmin())]}'))
    lines.append(f"{lab} & {pl['n']} & {f4(pl['acc'])} & {f4(pl['bal_acc'])} & {f4(pl['f1'])} & {f4(pl['auc_m'])} & {f4(pl['kappa'])} & "
                 f"{f4(pl['mcc'])} & {kk.mean():.4f}$\\pm${kk.std(ddof=1):.4f} & {kk.min():.4f} \\\\")
T1 = pd.DataFrame(rows); T1.to_csv(OUT / 'tables' / 'v2_protocols.csv', index=False); write('v2_protocols.tex', lines)

# T2: class-wise (strict and best calibrated)
for k in ['strict', 'strict_smooth'] + [f'calibrated_{cs}s' for cs in CFG['CAL_S']]:
    pl = RES[k][1]
    t = [f"{c}~({i}) & {f4(pl['prec_k'][i])} & {f4(pl['rec_k'][i])} & {f4(pl['f1_k'][i])} & {pl['sup_k'][i]} \\\\" for i, c in enumerate(CLASSES)]
    t += [f"Macro Avg & {f4(pl['prec'])} & {f4(pl['rec'])} & {f4(pl['f1'])} & {pl['n']} \\\\", r'\midrule']
    for lab, key in [('Accuracy', 'acc'), ('Balanced Acc.', 'bal_acc'), ('Specificity Macro', 'spec'), ('AUC-Macro', 'auc_m'),
                     ('AUC-Weighted', 'auc_w'), ("Cohen's $\\kappa$", 'kappa'), ('MCC', 'mcc'), ('$\\Delta=|$Acc$-$BalAcc$|$', 'gap')]:
        t.append(f"{lab} & \\multicolumn{{3}}{{c}}{{--}} & {f4(pl[key])} \\\\")
    write(f'v2_classwise_{k}.tex', t)

# T3: per subject (strict) with κ under the other protocols
cols = ['acc', 'prec', 'rec', 'f1', 'bal_acc', 'spec', 'auc_m', 'auc_w', 'kappa', 'mcc']
subj = RES['strict'][0]; t, M = [], []
for s in FOLDS:
    m = subj[s]; M.append([m[c] for c in cols])
    extra = ' & '.join(f"{RES[k][0][s]['kappa']:.4f}" for k in ['strict_smooth'] + [f'calibrated_{cs}s' for cs in CFG['CAL_S']])
    t.append(f"S{s} & {SEL[s]} & {'/'.join(map(str, m['sup_k']))} & " + ' & '.join(f4(m[c]) for c in cols) + f' & {extra} \\\\')
M = np.array(M, float)
t += [r'\midrule', 'Mean & -- & -- & ' + ' & '.join(f4(x) for x in np.nanmean(M, 0)) + ' & -- \\\\',
      'SD & -- & -- & ' + ' & '.join(f4(x) for x in np.nanstd(M, 0, ddof=1)) + ' & -- \\\\']
write('v2_subjects.tex', t)
pd.DataFrame([dict(subject=f'S{s}', subset=SEL[s], **{c: subj[s][c] for c in cols}) for s in FOLDS]).to_csv(OUT / 'tables' / 'v2_subjects.csv', index=False)

# T4: integer confusion matrices (strict) per subject + pooled
write('v2_confusion_strict.tex', [f"S{s} & " + ' & '.join(map(str, subj[s]['cm'].ravel())) + ' \\\\' for s in FOLDS] +
      [r'\midrule', 'Pooled & ' + ' & '.join(map(str, RES['strict'][1]['cm'].ravel())) + ' \\\\'])

# T5: scenarios (selected subset per fold)
sc_rows, sc_lines = [], []
for sc in CFG['SCENARIOS']:
    get = (lambda s, sc=sc: (lambda d: (d.y_true.values, d[['p0', 'p1', 'p2']].values))(load_preds(cfg_name(SEL[s], sc), s)))
    sj, pl = collect(get); kk = np.array([m['kappa'] for m in sj.values()])
    sc_rows.append(dict(scenario=sc, acc=pl['acc'], f1=pl['f1'], kappa=pl['kappa'], auc_m=pl['auc_m'], mcc=pl['mcc'],
                        subj_kappa_mean=kk.mean(), subj_kappa_sd=kk.std(ddof=1), worst_kappa=kk.min()))
SCN = pd.DataFrame(sc_rows); SCN['kappa_retained_%'] = 100 * SCN.kappa / SCN.kappa.iloc[0]
for _, r in SCN.iterrows():
    sc_lines.append(f"{r.scenario.capitalize()} & {r.acc:.4f} & {r.f1:.4f} & {r.auc_m:.4f} & {r.kappa:.4f} & {r.mcc:.4f} & "
                    f"{r.subj_kappa_mean:.4f}$\\pm${r.subj_kappa_sd:.4f} & {r.worst_kappa:.4f} & {r['kappa_retained_%']:.1f} \\\\")
SCN.to_csv(OUT / 'tables' / 'v2_scenarios.csv', index=False); write('v2_scenarios.tex', sc_lines)

# T6: reliability-bit screening (corrupted vs clean clients)
scr_rows = []
for sc in SCN_OTHER:
    fb = fc = nb = nc = 0
    for s in FOLDS:
        h = hist_of(cfg_name(SEL[s], sc), s, S0)['hist']; bad = set(h['corrupted'])
        for fl in h['flags'][CFG['WARMUP']:]:
            for cid, q in fl.items():
                if int(cid) in bad: nb += 1; fb += q
                else: nc += 1; fc += q
    scr_rows.append(dict(scenario=sc, flag_rate_corrupted=fb / max(1, nb), flag_rate_clean=fc / max(1, nc),
                         precision=fb / max(1, fb + fc)))
SCR = pd.DataFrame(scr_rows); SCR.to_csv(OUT / 'tables' / 'v2_screening.csv', index=False)
write('v2_screening.tex', [f"{r.scenario.capitalize()} & {r.flag_rate_corrupted:.3f} & {r.flag_rate_clean:.3f} & {r.precision:.3f} \\\\" for _, r in SCR.iterrows()])

# T7: sensor-subset sensitivity (reported AFTER the validation choice; never used to choose)
ss_rows, ss_lines = [], []
for key in SUBSET_KEYS:
    get = (lambda s, key=key: (lambda d: (d.y_true.values, d[['p0', 'p1', 'p2']].values))(load_preds(cfg_name(key), s)))
    sj, pl = collect(get); kk = np.array([m['kappa'] for m in sj.values()])
    vmean = np.mean([hist_of(cfg_name(key), s, S0)['best_val'] for s in FOLDS]); nsel = sum(SEL[s] == key for s in FOLDS)
    nch = len(chans(CFG['SUBSETS'][SUBSET_KEYS.index(key)]))
    ss_rows.append(dict(subset=key, n_ch=nch, val_f1=vmean, acc=pl['acc'], f1=pl['f1'], kappa=pl['kappa'], subj_kappa_sd=kk.std(ddof=1),
                        worst_kappa=kk.min(), times_selected=nsel))
    ss_lines.append(f"{key.replace('+', ' + ')} & {nch} & {vmean:.4f} & {pl['acc']:.4f} & {pl['f1']:.4f} & {pl['kappa']:.4f} & {kk.min():.4f} & {nsel}/{len(FOLDS)} \\\\")
pl = RES['strict'][1]; kk = np.array([m['kappa'] for m in RES['strict'][0].values()])
ss_lines += [r'\midrule', f"\\textbf{{Validation-selected per fold}} & {np.mean([len(chans(CFG['SUBSETS'][SUBSET_KEYS.index(SEL[s])])) for s in FOLDS]):.2f} & -- & "
             f"{pl['acc']:.4f} & {pl['f1']:.4f} & {pl['kappa']:.4f} & {kk.min():.4f} & -- \\\\"]
pd.DataFrame(ss_rows).to_csv(OUT / 'tables' / 'v2_subsets.csv', index=False); write('v2_subsets.tex', ss_lines)

# complexity
cx = []
for key, groups in zip(SUBSET_KEYS, CFG['SUBSETS']):
    m = TRACEH(len(chans(groups)), sum(g in groups for g in FEAT_GROUP)); cx.append(dict(subset=key, params=n_params(m), KB_fp32=4 * n_params(m) / 1024))
pd.DataFrame(cx).to_csv(OUT / 'tables' / 'v2_complexity.csv', index=False)

# seed statement + macros
(OUT / 'tables' / 'v2_seed_statement.tex').write_text(
    'All results come from a single training run per configuration (random seed 42). Reported standard deviations are computed '
    'across the 15 held-out subjects and describe between-subject variability; they are not estimates of seed-to-seed variability.\n')
def mac(name, v, fmt='{:.4f}'): return f'\\newcommand{{\\{name}}}{{{fmt.format(v)}}}'
pS, pM = RES['strict'][1], RES['strict_smooth'][1]
mlines = ['% AUTO-GENERATED by fedphysio_v2.ipynb',
          mac('VTwoAcc', 100 * pS['acc'], '{:.2f}'), mac('VTwoFone', pS['f1']), mac('VTwoKappa', pS['kappa']), mac('VTwoMCC', pS['mcc']),
          mac('VTwoAUC', pS['auc_m']), mac('VTwoBalAcc', pS['bal_acc']), mac('VTwoAmuRecall', pS['rec_k'][2]),
          f"\\newcommand{{\\VTwoNTest}}{{{pS['n']:,}}}".replace(',', '{,}'),
          mac('VTwoSmoothAcc', 100 * pM['acc'], '{:.2f}'), mac('VTwoSmoothKappa', pM['kappa']), mac('VTwoSmoothFone', pM['f1'])]
for cs in CFG['CAL_S']:
    pc, ps = RES[f'calibrated_{cs}s'][1], RES[f'strict_same_windows_{cs}s'][1]
    tag = {30: 'Thirty', 60: 'Sixty', 120: 'OneTwenty'}.get(cs, f'S{cs}')
    mlines += [mac(f'VTwoCal{tag}Acc', 100 * pc['acc'], '{:.2f}'), mac(f'VTwoCal{tag}Kappa', pc['kappa']), mac(f'VTwoCal{tag}Fone', pc['f1']),
               mac(f'VTwoSame{tag}Acc', 100 * ps['acc'], '{:.2f}')]
for _, r in SCN.iterrows():
    mlines.append(mac(f"VTwoKappa{r.scenario.capitalize()}", r.kappa))
for _, r in SCR.iterrows():
    mlines += [mac(f"VTwoFlagBad{r.scenario.capitalize()}", 100 * r.flag_rate_corrupted, '{:.1f}'),
               mac(f"VTwoFlagClean{r.scenario.capitalize()}", 100 * r.flag_rate_clean, '{:.1f}')]
(OUT / 'tables' / 'v2_results_macros.tex').write_text('\n'.join(mlines) + '\n'); print('\n'.join(mlines))

# %% [markdown]
# ## 14. Figures (editable: SVG with live text, PDF with TrueType fonts)

# %%
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 9, 'savefig.bbox': 'tight', 'figure.dpi': 150})
BLUE, ORANGE, RED, GREEN, PURPLE = '#4A90B8', '#F2A33A', '#E85D6A', '#3BA55C', '#7E57C2'
def savefig(fig, name):
    for ext in ['pdf', 'svg', 'png']: fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=300)
    plt.close(fig)

# confusion matrices for the protocols
prots = ['strict', 'strict_smooth'] + [f'calibrated_{cs}s' for cs in CFG['CAL_S']]
fig, axs = plt.subplots(1, len(prots), figsize=(3.3 * len(prots), 3.1))
for ax, k in zip(np.atleast_1d(axs), prots):
    cm = RES[k][1]['cm']; cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap='Greens', vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{cmn[i, j]:.2f}\n({cm[i, j]})', ha='center', va='center', fontsize=8, color='white' if cmn[i, j] > 0.6 else 'black')
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASSES, fontsize=7); ax.set_yticks(range(K)); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_title(PROT_LABEL[k], fontsize=8); ax.set_xlabel('Predicted')
np.atleast_1d(axs)[0].set_ylabel('True'); savefig(fig, 'v2_confusion_protocols')

# per-subject kappa by protocol
fig, ax = plt.subplots(figsize=(9, 3.4)); x = np.arange(len(FOLDS)); w = 0.8 / len(prots)
for i, (k, c) in enumerate(zip(prots, [BLUE, ORANGE, GREEN, PURPLE, RED])):
    ax.bar(x + (i - (len(prots) - 1) / 2) * w, [RES[k][0][s]['kappa'] for s in FOLDS], w, color=c, label=PROT_LABEL[k])
ax.set_xticks(x); ax.set_xticklabels([f'S{s}' for s in FOLDS]); ax.set_ylabel("Cohen's κ"); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)
ax.legend(fontsize=7, ncol=2, loc='lower right'); ax.grid(axis='y', alpha=0.3); savefig(fig, 'v2_subject_kappa')

# scenarios + screening
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].bar(SCN.scenario.str.capitalize(), SCN.kappa, color=[GREEN, ORANGE, RED][:len(SCN)])
for i, r in SCN.iterrows(): axs[0].text(i, r.kappa + 0.01, f'{r.kappa:.3f}\n({r["kappa_retained_%"]:.0f}%)', ha='center', fontsize=8)
axs[0].set_ylabel("Pooled Cohen's κ"); axs[0].set_ylim(0, 1.05); axs[0].set_title('(a) FedPhysio v2 under the three scenarios', fontsize=9)
xx = np.arange(len(SCR))
axs[1].bar(xx - 0.2, 100 * SCR.flag_rate_corrupted, 0.4, color=RED, label='Corrupted clients')
axs[1].bar(xx + 0.2, 100 * SCR.flag_rate_clean, 0.4, color=BLUE, label='Clean clients')
axs[1].set_xticks(xx); axs[1].set_xticklabels(SCR.scenario.str.capitalize()); axs[1].set_ylabel('Client-rounds flagged (%)')
axs[1].set_title('(b) Reliability bit', fontsize=9); axs[1].legend(fontsize=7); savefig(fig, 'v2_scenarios_screening')

# subset sensitivity
SS = pd.DataFrame(ss_rows)
fig, ax = plt.subplots(figsize=(9, 3.4)); x = np.arange(len(SS))
ax.bar(x - 0.2, SS.acc, 0.4, color=BLUE, label='Held-out accuracy'); ax.bar(x + 0.2, SS.f1, 0.4, color=ORANGE, label='Held-out macro-F1')
for i, r in SS.iterrows(): ax.text(i, max(r.acc, r.f1) + 0.01, f'{r.times_selected}/{len(FOLDS)}', ha='center', fontsize=7, color='#c0392b')
ax.set_xticks(x); ax.set_xticklabels([k.replace('+', '\n') for k in SS.subset], fontsize=7); ax.set_ylim(0, 1.05)
ax.set_title('Sensor subsets (held-out results reported after the validation choice; labels = folds selected)', fontsize=8)
ax.legend(fontsize=7); savefig(fig, 'v2_subsets')

# %% [markdown]
# ## 15. Summary: what to send back

# %%
print('=' * 90)
print('FedPhysio v2 — summary (send me the whole fedphysio_v2_results/tables folder and this printout)')
print('=' * 90)
print(T1[['protocol', 'n', 'acc', 'f1', 'kappa', 'auc_m', 'subj_kappa_mean', 'subj_kappa_sd', 'worst_kappa', 'worst_subject']].to_string(index=False))
print('\nScenarios:'); print(SCN.to_string(index=False))
print('\nScreening:'); print(SCR.to_string(index=False))
print('\nSubsets:'); print(pd.DataFrame(ss_rows).to_string(index=False))
r = CFG['REF_SUBMITTED']
print(f"\nReference (corrected FedPhysio v1): acc {r['acc']:.4f}, F1 {r['f1']:.4f}, kappa {r['kappa']:.4f}")
print(f"FedPhysio v2 strict LOSO:           acc {pS['acc']:.4f}, F1 {pS['f1']:.4f}, kappa {pS['kappa']:.4f}")
print('Consistency checks pass:', all(check(*v) for v in RES.values()))
