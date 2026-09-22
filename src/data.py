"""Data loading, non-IID partitioning, and drift transforms."""
import hashlib
from pathlib import Path
import wave

import numpy as np
import torch
import torch.nn.functional as F


SPEECHCOMMANDS35_LABELS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
)


def _load_cifar10(root, test_size, seed):
    from torchvision import datasets
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

    def to_tensor(ds):
        X = ds.data.astype(np.float32) / 255.0          # (N,32,32,3)
        X = (X - mean) / std
        X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # (N,3,32,32)
        y = torch.tensor(np.asarray(ds.targets), dtype=torch.long)
        return X, y

    Xtr, ytr = to_tensor(tr)
    Xte, yte = to_tensor(te)
    if test_size and test_size < len(yte):
        idx = np.random.default_rng(seed).choice(len(yte), test_size, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    return (Xtr, ytr), (Xte, yte), 10, (3, 32, 32)


def _load_synthetic(n_classes, n_features, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, size=(n_classes, n_features)).astype(np.float32)

    def gen(n):
        y = rng.integers(0, n_classes, size=n)
        X = centers[y] + rng.normal(0, 1, size=(n, n_features)).astype(np.float32)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    Xtr, ytr = gen(n_train)
    Xte, yte = gen(n_test)
    return (Xtr, ytr), (Xte, yte), n_classes, (n_features,)


def _speechcommand_logmel(waveform, sample_rate, mel_transform):
    """Convert one Speech Commands utterance to the CNN's 1x32x32 input."""
    if sample_rate != 16000:
        raise ValueError(f"Speech Commands sample rate must be 16000 Hz, got {sample_rate}")
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = F.pad(waveform[..., :16000], (0, max(0, 16000 - waveform.shape[-1])))
    feature = torch.log(mel_transform(waveform).clamp_min(1e-6))
    if feature.shape[-1] != 32:
        feature = F.interpolate(feature.unsqueeze(0), size=(32, 32), mode="bilinear",
                                align_corners=False).squeeze(0)
    return (feature - feature.mean()) / feature.std(unbiased=False).clamp_min(1e-6)


def _load_pcm16_wave(path):
    """Read the uncompressed 16-bit PCM used by Speech Commands."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError(f"expected uncompressed 16-bit PCM WAV: {path}")
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2",
        ).reshape(-1, channels).T.copy()
    return torch.from_numpy(samples).float().div_(32768.0), sample_rate


def _load_speechcommands35(root, test_size, seed, n_train):
    """Materialize a deterministic subset of the official v0.02 train/test split."""
    from torchaudio import datasets, transforms

    train = datasets.SPEECHCOMMANDS(root, download=True, subset="training")
    test = datasets.SPEECHCOMMANDS(root, download=False, subset="testing")
    n_test = len(test) if not test_size else int(test_size)
    if n_train > len(train) or n_test > len(test):
        raise ValueError(
            f"Speech Commands subset requests train={n_train}, test={n_test}; "
            f"available train={len(train)}, test={len(test)}"
        )

    rng = np.random.default_rng(seed)
    label_ids = {label: index for index, label in enumerate(SPEECHCOMMANDS35_LABELS)}
    mel = transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=512, n_mels=32,
    )

    def materialize(dataset, count):
        features, labels = [], []
        for index in rng.choice(len(dataset), count, replace=False):
            path = dataset._walker[int(index)]
            waveform, sample_rate = _load_pcm16_wave(path)
            label = Path(path).parent.name
            features.append(_speechcommand_logmel(waveform, sample_rate, mel))
            labels.append(label_ids[label])
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    Xtr, ytr = materialize(train, n_train)
    Xte, yte = materialize(test, n_test)
    return (Xtr, ytr), (Xte, yte), len(SPEECHCOMMANDS35_LABELS), (1, 32, 32)


def load_data(cfg):
    """Returns (Xtr,ytr),(Xte,yte), n_classes, input_shape."""
    if cfg.dataset == "cifar10":
        return _load_cifar10(cfg.data_root, cfg.test_size, cfg.seed)
    elif cfg.dataset == "speechcommands35":
        n_train = cfg.num_clients * cfg.samples_per_client
        return _load_speechcommands35(cfg.data_root, cfg.test_size, cfg.seed, n_train)
    elif cfg.dataset == "synthetic":
        n_train = cfg.num_clients * cfg.samples_per_client * 2
        return _load_synthetic(10, 20, n_train, cfg.test_size, cfg.seed)
    raise ValueError(f"unknown dataset {cfg.dataset}")


def partition_without_replacement(y, client_class_probs, samples_per_client, seed):
    """Allocate exact-size client partitions from one shuffled global index pool."""
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    n_classes = int(y_np.max()) + 1
    probs = np.asarray(client_class_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != n_classes or np.any(probs < 0):
        raise ValueError("client_class_probs has an invalid shape or value")
    num_clients = len(probs)
    requested = num_clients * int(samples_per_client)
    if requested > len(y_np):
        raise ValueError(f"partition requests {requested} unique samples from {len(y_np)}")
    pools = [rng.permutation(np.flatnonzero(y_np == c)) for c in range(n_classes)]
    cursors = np.zeros(n_classes, dtype=int)
    parts = [[] for _ in range(num_clients)]
    for _ in range(int(samples_per_client)):
        for ci in rng.permutation(num_clients):
            remaining = np.asarray([len(pools[c]) - cursors[c] for c in range(n_classes)])
            weights = probs[ci] * (remaining > 0)
            if weights.sum() <= 0:
                weights = remaining
            label = int(rng.choice(n_classes, p=weights / weights.sum()))
            parts[ci].append(int(pools[label][cursors[label]]))
            cursors[label] += 1
    return [np.asarray(part, dtype=np.int64) for part in parts]


def dirichlet_partition(y, num_clients, beta, samples_per_client, seed):
    """Split unique indices across clients with Dirichlet class skew (AC1.4.1)."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    n_classes = int(y.max()) + 1
    probs = np.random.default_rng(seed).dirichlet([beta] * n_classes, size=num_clients)
    return partition_without_replacement(y, probs, samples_per_client, seed + 1)


def iid_partition(y, num_clients, samples_per_client, seed):
    """Split a shuffled global sample uniformly into equal, non-overlapping clients."""
    requested = int(num_clients) * int(samples_per_client)
    if requested > len(y):
        raise ValueError(f"partition requests {requested} unique samples from {len(y)}")
    indices = np.random.default_rng(seed).permutation(len(y))[:requested]
    return [part.astype(np.int64) for part in np.split(indices, int(num_clients))]


def partition_sha256(parts, partition_mode=None):
    digest = hashlib.sha256()
    if partition_mode is not None:
        if partition_mode not in {"iid", "noniid"}:
            raise ValueError("invalid partition_mode for partition hash")
        digest.update(f"partition_mode:{partition_mode}\0".encode("ascii"))
    for part in parts:
        values = np.asarray(part, dtype="<i8")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def make_drift_map(n_classes, seed):
    """Deterministic label permutation = sudden real drift P(y|x) (AC1.2.6)."""
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(n_classes)
    # avoid accidental identity so the drift is always visible
    if np.array_equal(perm, np.arange(n_classes)):
        perm = np.roll(perm, 1)
    return torch.tensor(perm, dtype=torch.long)


def apply_virtual_shift(X, shift):
    """Virtual drift P(x): add a fixed constant to inputs. y|x semantics unchanged,
    only the input distribution moves -> a model that keeps its knowledge can adapt."""
    return X + shift


def resample_label_longtail(X, y, imbalance_factor, n_classes, seed, min_per_class=1):
    """Label drift P(y): resample to a long-tail class marginal without dropping
    any class that exists in the source split."""
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    idx_by_class = [np.where(y_np == c)[0] for c in range(n_classes)]
    available = [c for c, idx in enumerate(idx_by_class) if len(idx)]
    if not available:
        return X, y

    order = rng.permutation(n_classes)
    ranks = np.empty(n_classes, dtype=np.float32)
    ranks[order] = np.arange(n_classes, dtype=np.float32)
    factor = max(float(imbalance_factor), 1.0)
    denom = max(1, n_classes - 1)
    weights = factor ** (-ranks[available] / denom)
    probs = weights / weights.sum()

    n = len(y_np)
    min_count = max(0, int(min_per_class))
    if n < min_count * len(available):
        min_count = 0
    counts = np.zeros(n_classes, dtype=int)
    remaining = n - min_count * len(available)
    raw = probs * remaining
    base = np.floor(raw).astype(int)
    for c, b in zip(available, base):
        counts[c] = min_count + int(b)
    leftovers = n - int(counts.sum())
    if leftovers > 0:
        frac_order = np.argsort(raw - base)[::-1]
        for k in frac_order[:leftovers]:
            counts[available[int(k)]] += 1

    selected = []
    for c in available:
        take = int(counts[c])
        if take <= 0:
            continue
        pool = idx_by_class[c]
        selected.extend(rng.choice(pool, size=take, replace=len(pool) < take).tolist())
    rng.shuffle(selected)
    sel = torch.as_tensor(selected, dtype=torch.long)
    return X[sel], y[sel]


def make_drift_maps(n_concepts, n_classes, seed):
    """K distinct label permutations, one per concept (staggered drift: different
    clusters drift to different concepts, AC1.2.4). Reuses make_drift_map with an
    offset seed per concept. Returns {1: perm1, 2: perm2, ...}."""
    return {cid: make_drift_map(n_classes, seed + 1000 * cid)
            for cid in range(1, n_concepts + 1)}
