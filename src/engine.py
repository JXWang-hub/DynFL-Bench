"""FL engine. Action space includes no_op | lr_reset | robust | dropout_handle |
friend_substitute | fedprox | spawn_concept | spawn_concept_auto |
label_prior_adapt | reweight.

  - lr_reset       restore LR to lr0 (adapt; correct for genuine drift)
  - robust         persistent trimmed-mean aggregation (reject faulty clients)
  - dropout_handle stale-cache dropout baseline
  - friend_substitute use the current update of the most label-similar online friend

Evaluation (对症):
  - scenario "drift": after drift_round, eval on NEW concept (follow drift)
  - scenario "fault"/"dropout": eval on ORIGINAL task
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from telemetry import clustering_metrics, collect, norm_stats
from observation_contract import (
    private_scorer_record, public_observation, serialize_public_observation,
)
from runtime_contract import (ClientRuntimeState, FedDAAReport, FedDriftReport,
                              InputShiftReport, LabelHistogramReport, LabelShiftReport,
                              UpdateEnvelope)
from episode_timeline import compile_episode_timeline
from action_contract import (
    ACTION_SPECS, Action, ActionBundle, ActionState, as_action_bundle,
    reduce_action_state,
)


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def build_model(n_classes, input_shape):
    if len(input_shape) == 3:
        return SmallCNN(input_shape[0], n_classes)
    return SmallMLP(input_shape[0], n_classes)


def _scaled_sign_flip(state, base, scale):
    """Replace a local delta with -scale * delta around the global state."""
    return {
        key: (base[key].detach().cpu() - scale *
              (value - base[key].detach().cpu())).to(value.dtype)
        for key, value in state.items()
    }


def local_train(model, Xc, yc, cfg, lr, device, prox_mu=0.0, global_params=None,
                class_weight=None, logit_calibration=None):
    """SGD local training. With prox_mu>0 adds the FedProx proximal term
    (mu/2)||w - w_global||^2 -> constrains client drift (set_aggregator(fedprox))."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(yc)
    idx = np.arange(n)
    losses = []
    for _ in range(cfg.local_epochs):
        np.random.shuffle(idx)
        for s in range(0, n, cfg.batch_size):
            b = idx[s:s + cfg.batch_size]
            xb = Xc[b].to(device)
            yb = yc[b].to(device)
            opt.zero_grad()
            logits = model(xb)
            if logit_calibration is not None:
                logits = logits - logit_calibration.to(device)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            if prox_mu > 0 and global_params is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + (prox_mu / 2) * prox
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def aggregate(global_model, client_states, weights):
    total = float(sum(weights))
    new = {}
    for k in global_model.state_dict().keys():
        new[k] = sum(w * cs[k].float() for cs, w in zip(client_states, weights)) / total
    global_model.load_state_dict(new)


def aggregate_robust(global_model, client_states, trim):
    keys = global_model.state_dict().keys()
    c = len(client_states)
    k = int(c * trim)
    new = {}
    for key in keys:
        stacked = torch.stack([cs[key].float() for cs in client_states], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if k > 0 and c > 2 * k:
            sorted_vals = sorted_vals[k:c - k]
        new[key] = sorted_vals.mean(dim=0)
    global_model.load_state_dict(new)


def _flat_state(st):
    return torch.cat([st[k].float().flatten() for k in st])


def _pairwise_state_distance(client_states):
    vecs = torch.stack([_flat_state(st) for st in client_states], dim=0)
    diff = vecs[:, None, :] - vecs[None, :, :]
    return (diff * diff).sum(dim=2)


def _distance_trimmed_indices(client_states, compromised_num):
    c = len(client_states)
    if c == 0:
        return [], 0
    actual = max(0, min(int(compromised_num), (c - 1) // 2))
    keep = max(1, c - 2 * actual)
    dist_sum = _pairwise_state_distance(client_states).sum(dim=1)
    median = dist_sum.median()
    chosen = torch.argsort((dist_sum - median).abs())[:keep]
    return [int(i) for i in chosen], actual


def aggregate_distance_trimmed_mean(global_model, client_states, weights, compromised_num):
    chosen, actual = _distance_trimmed_indices(client_states, compromised_num)
    aggregate(global_model, [client_states[i] for i in chosen], [weights[i] for i in chosen])
    return {"compromised_num": actual, "keep_n": len(chosen), "selected": chosen}


def _update_sketch(st, gstate, limit=512):
    """Small update-vector sketch for cheap client clustering."""
    chunks, seen = [], 0
    for k in st:
        v = (st[k] - gstate[k].cpu()).float().flatten()
        take = min(limit - seen, v.numel())
        if take > 0:
            chunks.append(v[:take])
            seen += take
        if seen >= limit:
            break
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    out = torch.cat(chunks).numpy()
    norm = np.linalg.norm(out)
    return out / norm if norm > 1e-12 else out


def _cosine_distance(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - np.dot(a, b) / denom) if denom > 1e-12 else 1.0


def _feddrift_cluster_clients(features, cfg):
    """FedDrift-style tiny agglomerative clustering over update sketches."""
    clients = sorted(features)
    if not clients:
        return {}, 0.0
    clusters = [[ci] for ci in clients]

    def cdist(a, b):
        return float(np.mean([_cosine_distance(features[i], features[j])
                              for i in a for j in b]))

    while len(clusters) > 1:
        best = min(((cdist(clusters[i], clusters[j]), i, j)
                    for i in range(len(clusters)) for j in range(i + 1, len(clusters))),
                   key=lambda x: x[0])
        must_merge = len(clusters) > int(cfg.feddrift_max_concepts)
        if not must_merge and best[0] > float(cfg.feddrift_cluster_threshold):
            break
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)

    min_size = max(1, int(cfg.feddrift_min_cluster_size))
    large = [c for c in clusters if len(c) >= min_size]
    small = [c for c in clusters if len(c) < min_size]
    if large:
        for c in small:
            target = min(range(len(large)), key=lambda i: cdist(c, large[i]))
            large[target].extend(c)
        clusters = large

    assignment = {ci: lab + 1 for lab, c in enumerate(clusters) for ci in c}
    split = 0.0
    if len(clusters) > 1:
        split = min(cdist(clusters[i], clusters[j])
                    for i in range(len(clusters)) for j in range(i + 1, len(clusters)))
    return assignment, split


def _feddrift_align_assignment(new_assignment, old_assignment, next_id):
    if not old_assignment:
        return new_assignment, max(new_assignment.values(), default=0) + 1
    out, used = {}, set()
    for raw in sorted(set(new_assignment.values())):
        members = [ci for ci, lab in new_assignment.items() if lab == raw]
        counts = {}
        for ci in members:
            if ci in old_assignment:
                counts[old_assignment[ci]] = counts.get(old_assignment[ci], 0) + 1
        label = max(counts, key=counts.get) if counts else next_id
        if label in used:
            label = next_id
        if label >= next_id:
            next_id = label + 1
        used.add(label)
        for ci in members:
            out[ci] = label
    return out, next_id


def _feddrift_cluster_report(assignment, concept_of):
    if not assignment or not concept_of:
        return "n/a"
    total, correct = 0, 0
    for lab in set(assignment.values()):
        members = [ci for ci, v in assignment.items() if v == lab and ci in concept_of]
        if not members:
            continue
        counts = {}
        for ci in members:
            counts[concept_of[ci]] = counts.get(concept_of[ci], 0) + 1
        correct += max(counts.values())
        total += len(members)
    return round(correct / total, 4) if total else "n/a"


def _feddrift_update_detector(ref, ema, confirm, ci, loss, cfg):
    if np.isnan(ref[ci]):
        ref[ci] = loss
        ema[ci] = loss
        confirm[ci] = 0
        return 0.0
    ema[ci] = 0.8 * float(ema[ci]) + 0.2 * float(loss)
    score = float(loss) - float(ref[ci])
    if score > max(float(cfg.feddrift_loss_delta), 0.15):
        confirm[ci] += 1
    else:
        confirm[ci] = 0
    return score


def _feddrift_update_reports(reports, ref_acc, confirm, scores, cfg):
    """Update server drift state from current client reports only."""
    client_ids = [report.client_id for report in reports]
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("duplicate FedDrift client report")
    confirmed, maximum = [], 0.0
    threshold = float(getattr(cfg, "feddrift_h_delta", cfg.feddrift_loss_delta))
    required = max(1, int(cfg.feddrift_confirm_rounds))
    for report in reports:
        best = max(report.scores().values())
        ci = report.client_id
        score = 0.0 if np.isnan(ref_acc[ci]) else float(ref_acc[ci] - best)
        scores[ci] = score
        maximum = max(maximum, score)
        if report.source_round < int(getattr(cfg, "feddrift_warmup_rounds", 0)):
            confirm[ci] = 0
            ref_acc[ci] = best
        elif score > threshold:
            confirm[ci] += 1
        else:
            confirm[ci] = 0
            ref_acc[ci] = best if np.isnan(ref_acc[ci]) else 0.9 * ref_acc[ci] + 0.1 * best
        if confirm[ci] >= required:
            confirmed.append(ci)
    return confirmed, maximum


def _feddrift_stable_groups(confirmed, features, cfg):
    min_size = max(1, int(cfg.feddrift_min_cluster_size))
    usable = {ci: features[ci] for ci in confirmed if ci in features}
    if len(usable) < min_size:
        return []
    assignment, _ = _feddrift_cluster_clients(usable, cfg)
    groups = [[ci for ci, label in assignment.items() if label == cluster]
              for cluster in sorted(set(assignment.values()))]
    return sorted((group for group in groups if len(group) >= min_size),
                  key=lambda group: (-len(group), group))


def _feddrift_match_models(reports):
    """Match each client to the best broadcast model without scorer truth."""
    return {report.client_id: max(report.scores(), key=report.scores().get)
            for report in reports}


def _flat_update(st, gstate):
    return torch.cat([(st[k] - gstate[k].cpu()).float().flatten() for k in st])


def _fdms_pairwise_similarity(states_by_client, gstate):
    """FL-FDMS friend discovery signal: cosine of current model updates."""
    vecs = {ci: _flat_update(st, gstate) for ci, (st, _) in states_by_client.items()}
    sims = {}
    for i, vi in vecs.items():
        for j, vj in vecs.items():
            denom = float(vi.norm().item() * vj.norm().item())
            cos = float(torch.dot(vi, vj).item() / denom) if denom > 1e-12 else 0.0
            sims[(i, j)] = 0.5 * (cos + 1.0)
    return sims


def _fdms_update_matrix(M, T, sims):
    for (i, j), sim in sims.items():
        t = T[i, j]
        M[i, j] = (t * M[i, j] + sim) / (t + 1.0)
        T[i, j] = t + 1.0


def _fdms_substitution_error(real_state, friend_state, gstate):
    real = _flat_update(real_state, gstate)
    subst = _flat_update(friend_state, gstate)
    denom = float(real.norm().item())
    return float((real - subst).norm().item() / denom) if denom > 1e-12 else 0.0


def _init_oort_state(sizes, cfg):
    rng = np.random.default_rng(cfg.seed + 7100)
    size_scale = sizes / max(1.0, float(np.mean(sizes)))
    duration = 100.0 * size_scale * rng.uniform(0.6, 1.8, len(sizes))
    return {
        "reward": np.sqrt(np.maximum(1.0, sizes)).astype(float),
        "duration": duration.astype(float),
        "count": np.zeros(len(sizes), dtype=int),
        "last": np.full(len(sizes), -1, dtype=int),
        "exploration": float(cfg.oort_exploration_factor),
        "round_threshold": float(cfg.oort_round_threshold),
        "exploit_history": [],
    }


def _oort_update_client(state, ci, reward, round_idx):
    state["reward"][ci] = max(1e-6, float(reward))
    state["count"][ci] += 1
    state["last"][ci] = int(round_idx)


def _oort_blacklist(state, cfg):
    if cfg.oort_blacklist_rounds <= 0:
        return set()
    over = [i for i, c in enumerate(state["count"]) if c > cfg.oort_blacklist_rounds]
    cap = int(cfg.oort_blacklist_max_len * len(state["count"]))
    return set(sorted(over, key=lambda i: state["count"][i], reverse=True)[:cap])


def _oort_pacer(state, cfg):
    step = max(1, int(cfg.oort_pacer_step))
    hist = state["exploit_history"]
    if len(hist) < 2 * step:
        return
    prev = float(np.mean(hist[-2 * step:-step]))
    cur = float(np.mean(hist[-step:]))
    flat = abs(cur - prev) / max(abs(prev), 1e-12) < 0.01
    delta = cfg.oort_pacer_delta if flat else -cfg.oort_pacer_delta
    state["round_threshold"] = float(np.clip(state["round_threshold"] + delta, 1.0, 100.0))


def _oort_scores(state, sizes, losses, coverage, round_idx, cfg):
    blacklist = _oort_blacklist(state, cfg)
    rewards = np.asarray(state["reward"], dtype=float)
    hi = float(np.quantile(rewards, cfg.oort_clip_bound))
    lo = float(rewards.min())
    clipped = np.minimum(rewards, hi)
    norm = np.ones_like(clipped) if hi <= lo else (clipped - lo) / (hi - lo)
    feasible = [i for i in range(len(sizes)) if i not in blacklist]
    prefer = float(np.percentile(state["duration"][feasible], state["round_threshold"])) if feasible else 0.0
    out = []
    for ci in range(len(sizes)):
        age = round_idx - state["last"][ci] if state["last"][ci] >= 0 else round_idx + 1
        uncertainty = float(np.sqrt(0.1 * np.log(max(2, round_idx + 2)) * max(1, age) /
                                    max(1, state["count"][ci])))
        dur = float(state["duration"][ci])
        penalty = (prefer / dur) ** cfg.oort_round_penalty if prefer > 0 and dur > prefer else 1.0
        score = float((norm[ci] + uncertainty) * penalty)
        out.append({"client": ci, "loss": round(float(losses[ci]), 6),
                    "data_coverage": round(float(coverage[ci]), 6),
                    "availability": round(float(state["count"][ci] / max(1, round_idx + 1)), 6),
                    "utility": round(score, 6),
                    "oort_reward": round(float(rewards[ci]), 6),
                    "oort_reward_norm": round(float(norm[ci]), 6),
                    "oort_uncertainty": round(uncertainty, 6),
                    "oort_duration": round(dur, 6),
                    "oort_duration_penalty": round(float(penalty), 6),
                    "oort_score": round(score, 6),
                    "oort_unexplored": bool(state["count"][ci] == 0),
                    "oort_blacklisted": bool(ci in blacklist)})
    return out, prefer


def _model_loss(model, Xc, yc, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in range(0, len(yc), 256):
            xb = Xc[s:s + 256].to(device)
            yb = yc[s:s + 256].to(device)
            losses.append(F.cross_entropy(model(xb), yb, reduction="sum").item())
    return float(sum(losses) / max(1, len(yc)))


def _model_acc(model, Xc, yc, device, max_samples=0):
    if max_samples and len(yc) > max_samples:
        Xc = Xc[:max_samples]
        yc = yc[:max_samples]
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(yc), 256):
            xb = Xc[s:s + 256].to(device)
            yb = yc[s:s + 256].to(device)
            correct += int((model(xb).argmax(1) == yb).sum().item())
    return float(correct / max(1, len(yc)))


def _merge_model_weights(dst, src, dst_weight, src_weight):
    total = max(float(dst_weight) + float(src_weight), 1e-12)
    sd_dst = dst.state_dict()
    sd_src = src.state_dict()
    merged = {k: (sd_dst[k].detach() * float(dst_weight) +
                  sd_src[k].detach() * float(src_weight)) / total
              for k in sd_dst}
    dst.load_state_dict(merged)


def _softmax_neg(values, temp=1.0):
    vals = -np.asarray(values, dtype=float) / max(float(temp), 1e-6)
    vals -= vals.max()
    exp = np.exp(vals)
    return exp / max(float(exp.sum()), 1e-12)


def _softmax(values):
    vals = np.asarray(values, dtype=float)
    vals -= vals.max()
    exp = np.exp(vals)
    return exp / max(float(exp.sum()), 1e-12)


def _aggregate_label_reports(reports, n_classes, round_idx):
    client_ids = [report.client_id for report in reports]
    if (not reports or len(client_ids) != len(set(client_ids)) or
            any(report.source_round != round_idx or
                len(report.class_counts) != n_classes for report in reports)):
        raise ValueError("label histogram report mismatch")
    counts = np.sum([report.class_counts for report in reports], axis=0, dtype=float)
    return counts / max(float(counts.sum()), 1.0)


def _simulated_secure_label_shift(reports, n_classes, round_idx):
    """Aggregate roster-matched reports without claiming cryptographic security."""
    if not reports:
        return 0.0
    client_ids = [report.client_id for report in reports]
    if (len(client_ids) != len(set(client_ids)) or
            any(report.source_round != round_idx or
                len(report.baseline_counts) != n_classes or
                len(report.current_counts) != n_classes for report in reports)):
        raise ValueError("label shift report mismatch")
    baseline = np.sum([report.baseline_counts for report in reports], axis=0, dtype=float)
    current = np.sum([report.current_counts for report in reports], axis=0, dtype=float)
    baseline /= max(float(baseline.sum()), 1.0)
    current /= max(float(current.sum()), 1.0)
    return float(0.5 * np.abs(current - baseline).sum())


def _simulated_secure_input_shift(reports, round_idx):
    """Roster-matched, sample-weighted input mean delta after secure aggregation."""
    if not reports:
        return 0.0
    client_ids = [report.client_id for report in reports]
    if (len(client_ids) != len(set(client_ids)) or
            any(report.source_round != round_idx for report in reports)):
        raise ValueError("input shift report mismatch")
    baseline_mean = sum(report.baseline_sum for report in reports) / sum(
        report.baseline_count for report in reports
    )
    current_mean = sum(report.current_sum for report in reports) / sum(
        report.current_count for report in reports
    )
    return float(abs(current_mean - baseline_mean))


def _model_proto(model, Xc, yc, n_classes, device):
    model.eval()
    sums = torch.zeros((n_classes, n_classes), dtype=torch.float32)
    counts = torch.zeros(n_classes, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, len(Xc), 256):
            probs = F.softmax(model(Xc[s:s + 256].to(device)), dim=1).cpu()
            labels = yc[s:s + 256].cpu()
            sums.index_add_(0, labels, probs)
            counts.index_add_(0, labels, torch.ones(len(labels)))
    return (sums / counts[:, None].clamp_min(1.0)).numpy()


def _kmeans_small(X, k, seed, iters=20):
    X = np.asarray(X, dtype=float)
    if len(X) <= k:
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    centers = [X[int(rng.integers(0, len(X)))]]
    while len(centers) < k:
        d = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(X[int(np.argmax(d))])
    centers = np.asarray(centers)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new_labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cid in range(k):
            if np.any(labels == cid):
                centers[cid] = X[labels == cid].mean(axis=0)
    return labels


def _silhouette_small(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    if len(uniq) < 2 or len(uniq) >= len(X):
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    vals = []
    for i, lab in enumerate(labels):
        same = [j for j in range(len(X)) if labels[j] == lab and j != i]
        a = float(np.mean(D[i, same])) if same else 0.0
        b = min(float(np.mean(D[i, [j for j in range(len(X)) if labels[j] == other]]))
                for other in uniq if other != lab)
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals))


def _feddaa_auto_assignment(prototypes, cfg, seed):
    clients = sorted(prototypes)
    if not clients:
        return {}, 0.0, []
    X = np.asarray([prototypes[ci] for ci in clients], dtype=float)
    max_k = min(int(cfg.feddaa_max_clusters), len(clients) // max(1, int(cfg.feddaa_min_cluster_size)))
    baseline = ({ci: 0 for ci in clients}, 0.0, [X.mean(axis=0)])
    if max_k < 2:
        return baseline
    best_labels, best_score = None, -1.0
    for k in range(2, max_k + 1):
        labels = _kmeans_small(X, k, seed + k)
        if min(np.bincount(labels, minlength=k)) < int(cfg.feddaa_min_cluster_size):
            continue
        score = _silhouette_small(X, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    if best_labels is None or best_score < float(cfg.feddaa_tol_split):
        return baseline
    centers = [X[best_labels == lab].mean(axis=0) for lab in sorted(set(best_labels.tolist()))]
    remap = {lab: i for i, lab in enumerate(sorted(set(best_labels.tolist())))}
    return {ci: remap[int(lab)] for ci, lab in zip(clients, best_labels)}, float(best_score), centers


def _feddaa_recluster(features, old_assignment, cfg):
    assignment, score, _ = _feddaa_auto_assignment(features, cfg, 0)
    old_k = len(set((old_assignment or {}).values()))
    new_k = len(set(assignment.values()))
    event = "split" if new_k > old_k else "merge" if new_k < old_k else "none"
    return assignment, score, event


def _feddaa_rebuild_due(round_idx, last_rebuild_round, interval):
    return (last_rebuild_round is None or
            round_idx - last_rebuild_round >= max(1, int(interval)))


def _update_label_history(history, weights, reports, momentum):
    if not reports:
        return history
    k = history.shape[0]
    fresh = np.zeros_like(history)
    mass = np.zeros(k, dtype=float)
    for ci, report in reports.items():
        hist = np.asarray(report.class_counts, dtype=float) / report.sample_count
        for cid in range(k):
            w = float(weights[ci, cid])
            fresh[cid] += w * hist
            mass[cid] += w
    for cid in range(k):
        if mass[cid] > 1e-12:
            fresh[cid] /= mass[cid]
            history[cid] = momentum * history[cid] + (1.0 - momentum) * fresh[cid]
            history[cid] /= max(float(history[cid].sum()), 1e-12)
    return history


class Engine:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.timeline = compile_episode_timeline(cfg, data)
        self._timeline_cfg_key = repr(vars(cfg))

    def _b3_plan(self):
        return self.data.get("b3_episode_plan")

    def _timeline(self):
        key = repr(vars(self.cfg))
        if not hasattr(self, "timeline") or getattr(self, "_timeline_cfg_key", None) != key:
            self.timeline = compile_episode_timeline(self.cfg, self.data)
            self._timeline_cfg_key = key
        return self.timeline

    def _has_cause(self, cause):
        return self._timeline().has(cause)

    def _event_round(self):
        return self._timeline().event_round()

    def _moment_align(self, X):
        base_mean = float(self.data.get("baseline_xmean", 0.0))
        base_std = max(float(self.data.get("baseline_xstd", 1.0)), 1e-6)
        cur_mean = self.data.get("vshift_xmean")
        cur_std = self.data.get("vshift_xstd")
        if cur_mean is None or cur_std is None:
            xf = X.float()
            cur_mean = float(xf.mean())
            cur_std = float(xf.std(unbiased=False))
        cur_mean = float(cur_mean)
        cur_std = max(float(cur_std), 1e-6)
        return (X - cur_mean) / cur_std * base_std + base_mean

    def _predict(self, model, Xte=None):
        """Forward the whole test set, return predicted labels (cpu)."""
        Xte = self.data["Xte"] if Xte is None else Xte
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xte), 256):
                logits = model(Xte[s:s + 256].to(self.device))
                preds.append(logits.argmax(1).cpu())
        return torch.cat(preds)

    def _train_client(self, local, gstate, Xc, yc, lr, prox_mu=0.0,
                      class_weight=None, logit_calibration=None,
                      baseline_inputs=None, baseline_labels=None,
                      byzantine_scale=0.0):
        """Client-local train step plus the fixed-size report allowed to reach Engine."""
        local.load_state_dict(gstate)
        gparams = [p.detach().clone() for p in local.parameters()] if prox_mu > 0 else None
        loss = local_train(local, Xc, yc, self.cfg, lr, self.device, prox_mu, gparams,
                           class_weight, logit_calibration)
        st = {k: v.detach().cpu() for k, v in local.state_dict().items()}
        un = sum(((st[k] - gstate[k].cpu()).float() ** 2).sum().item() for k in st) ** 0.5
        if byzantine_scale > 0.0:
            st = _scaled_sign_flip(st, gstate, byzantine_scale)
            un *= byzantine_scale
        baseline_inputs = Xc if baseline_inputs is None else baseline_inputs
        report = {
            "input_sum": float(Xc.float().sum()),
            "input_count": int(Xc.numel()),
            "baseline_input_sum": float(baseline_inputs.float().sum()),
            "baseline_input_count": int(baseline_inputs.numel()),
            "label_counts": torch.bincount(
                yc.cpu(), minlength=self.data["n_classes"]
            ).tolist(),
            "baseline_label_counts": torch.bincount(
                (yc if baseline_labels is None else baseline_labels).cpu(),
                minlength=self.data["n_classes"],
            ).tolist(),
        }
        return st, un, loss, report

    def evaluate(self, model, r, moment_align=False):
        cfg = self.cfg
        data_shift = any(self._has_cause(cause) for cause in
                         ("real_drift", "virtual_drift", "label_prior_drift"))
        event_round = self._event_round()
        data_cause = next((cause for cause in
                           ("real_drift", "virtual_drift", "label_prior_drift")
                           if self._has_cause(cause)), None)
        client_ids = (range(len(self.data["client_X"])) if "client_X" in self.data
                      else self.data.get("drift_schedule", ()))
        progress = (self._timeline().active_fraction(data_cause, r, client_ids)
                    if data_shift else 0.0)
        yte, Xte = self.data["yte"], self.data["Xte"]

        def accuracy(X, y):
            correct = (self._predict(model, X) == y).float()
            per_class = []
            for c in range(self.data["n_classes"]):
                mask = (y == c)
                per_class.append(float(correct[mask].mean().item()) if mask.any() else None)
            return float(correct.mean().item()), per_class

        def blended_accuracy(new_X, new_y):
            new, new_classes = accuracy(new_X, new_y)
            if progress >= 1.0:
                return new, new_classes
            old, old_classes = accuracy(Xte, yte)
            classes = [b if a is None else a if b is None else (1.0 - progress) * a + progress * b
                       for a, b in zip(old_classes, new_classes)]
            return (1.0 - progress) * old + progress * new, classes

        if progress <= 0:
            overall, per_class = accuracy(Xte, yte)
        elif self._has_cause("virtual_drift"):
            shifted_X = self.data["vshift_Xte"]
            if moment_align:
                shifted_X = self._moment_align(shifted_X)
            overall, per_class = blended_accuracy(shifted_X, yte)
        elif self._has_cause("label_prior_drift"):
            overall, per_class = blended_accuracy(self.data["label_Xte"], self.data["label_yte"])
        else:
            overall, per_class = blended_accuracy(Xte, self.data["drift_map"][yte])
        return overall, min(value for value in per_class if value is not None)

    def evaluate_staggered(self, single_model, spawn_models, round_idx,
                           auto=False, assignment=None):
        """Evaluate only concepts active for clients at the current round."""
        yte = self.data["yte"]
        drift_maps = self.data["drift_maps"]
        concept_of = self.data.get("concept_of", {})
        schedule = self.data.get("drift_schedule", {})
        current = {
            ci: concept_of[ci] if round_idx >= schedule.get(ci, 10**9) else 0
            for ci in concept_of
        }
        accs, weights = [], []
        for cid in sorted(set(current.values())):
            if spawn_models and auto and assignment:
                counts = {}
                for ci, lab in assignment.items():
                    if current.get(ci) == cid:
                        counts[lab] = counts.get(lab, 0) + 1
                m = spawn_models.get(max(counts, key=counts.get)) if counts else None
                models = [m] if m is not None else list(spawn_models.values())
            elif spawn_models:
                model = spawn_models.get(cid)
                models = [model if model is not None else spawn_models[min(spawn_models)]]
            else:
                models = [single_model]
            target = yte if cid == 0 else drift_maps[cid][yte]
            accs.append(max(float((self._predict(m) == target).float().mean().item())
                             for m in models))
            weights.append(sum(value == cid for value in current.values()))
        return (float(np.average(accs, weights=weights)), min(accs))

    def present_clients(self, r, override=None):
        n = len(self.data["client_X"])
        cfg = self.cfg
        plan = self._b3_plan()
        if plan:
            invalid = 0
            if override is not None:
                seen, planned = set(), []
                for raw in override:
                    try:
                        ci = int(raw)
                    except (TypeError, ValueError):
                        invalid += 1
                        continue
                    if ci in seen or not 0 <= ci < n:
                        invalid += 1
                        continue
                    seen.add(ci); planned.append(ci)
                k = max(1, int(cfg.participation * n)) if cfg.participation < 1.0 else n
                planned = planned[:k]
            elif cfg.participation < 1.0:
                k = max(1, int(cfg.participation * n))
                rng = np.random.default_rng(cfg.seed + 5000 + r)
                planned = sorted(rng.choice(n, k, replace=False).tolist())
            else:
                planned = list(range(n))
            available = [ci for ci in range(n)
                         if not self._timeline().active("dropout", r, ci)]
            allowed = set(available)
            available_planned = [ci for ci in planned if ci in allowed]
            participating = list(available_planned)
            self.last_select_invalid = invalid
            self.last_select_used = override is not None and bool(planned)
            self.last_planned_clients = planned
            self.last_available_clients = available
            self.last_available_planned_clients = available_planned
            self.last_participating_clients = participating
            return participating
        invalid = 0
        in_window = self._timeline().active("dropout", r)
        available = list(range(n))
        if cfg.scenario == "dropout" and in_window:
            available = [ci for ci in range(n)
                         if not self._timeline().active("dropout", r, ci)]
        if override is not None:
            seen, chosen = set(), []
            allowed = set(available)
            for raw in override:
                try:
                    ci = int(raw)
                except (TypeError, ValueError):
                    invalid += 1; continue
                if ci in seen or ci not in allowed:
                    invalid += 1; continue
                seen.add(ci); chosen.append(ci)
            k = max(1, int(cfg.participation * n)) if cfg.participation < 1.0 else len(available)
            self.last_select_invalid = invalid
            self.last_select_used = bool(chosen)
            if chosen:
                return chosen[:k]
        self.last_select_invalid = 0
        self.last_select_used = False
        if cfg.scenario == "dropout" and in_window:
            return available
        if cfg.participation < 1.0:
            # partial participation: sample a fraction each round, keyed by (seed, round)
            # so every agent faces the same participating set (fair) and it's reproducible.
            k = max(1, int(cfg.participation * n))
            rng = np.random.default_rng(cfg.seed + 5000 + r)
            return sorted(rng.choice(n, k, replace=False).tolist())
        return list(range(n))

    def client_utilities(self, last_loss, seen_rounds, coverage, sizes, r, oort_state):
        return _oort_scores(oort_state, sizes, last_loss, coverage, r, self.cfg)

    def friend_for(self, ci, present, M=None, ready=False):
        """Most update-similar online client from FL-FDMS's history matrix."""
        if not ready or M is None or not present:
            return None, 0.0, 0
        best, best_sim = None, -1.0
        for pj in present:
            sim = float(M[ci, pj])
            if sim > best_sim:
                best, best_sim = pj, sim
        return best, best_sim, len(present)

    def infer_concept_clusters(self, features, old_assignment=None, next_id=1):
        assigned, split = _feddrift_cluster_clients(features, self.cfg)
        return _feddrift_align_assignment(assigned, old_assignment or {}, next_id) + (split,)

    def run(self, agent, log=True):
        cfg = self.cfg
        self.current_round = None
        device = self.device
        model = build_model(self.data["n_classes"], self.data["input_shape"]).to(device)
        local = build_model(self.data["n_classes"], self.data["input_shape"]).to(device)
        cx, cy = self.data["client_X"], self.data["client_y"]
        drift_clients = self.data["drift_clients"]
        fault_clients = self.data.get("fault_clients", drift_clients if cfg.scenario == "fault" else set())
        # ci -> round it starts using drifted labels (sudden: all == drift_round;
        # gradual: staggered in waves). Absent clients never flip.
        schedule = self.data.get("drift_schedule",
                                 {ci: cfg.drift_round for ci in drift_clients})
        drift_map = self.data["drift_map"]
        b3_plan = self._b3_plan()
        event_round = self._event_round()
        data_cause = next((cause for cause in
                           ("real_drift", "virtual_drift", "label_prior_drift")
                           if self._has_cause(cause)), None)
        staggered = (cfg.scenario == "staggered")
        concept_of = self.data.get("concept_of", {})    # ci -> concept id (1-based)
        drift_maps = self.data.get("drift_maps", {})     # cid -> label permutation
        n = len(cx)
        client_sizes = np.asarray([len(y) for y in cy], dtype=float)
        client_coverage = np.asarray([
            float((torch.bincount(y, minlength=self.data["n_classes"]) > 0).float().mean().item())
            for y in cy
        ], dtype=float)
        client_last_loss = np.ones(n, dtype=float)
        client_seen_rounds = np.zeros(n, dtype=float)
        oort_state = _init_oort_state(client_sizes, cfg)

        current_lr = cfg.lr0
        action_state = ActionState()
        use_robust = False
        use_dropout_handle = False
        use_friend_substitute = False
        use_fedprox = False
        use_reweight = False
        use_label_prior_adapt = False
        use_spawn = False
        spawn_auto = False
        use_drift_adapt = False
        use_retain_history_adapt = False
        use_moment_align = False
        feddaa_enabled = False
        feddaa_mode = ""
        feddaa_models = []
        feddaa_weights = np.ones((n, 1), dtype=float)
        feddaa_label_history = np.ones((1, self.data["n_classes"]), dtype=float) / self.data["n_classes"]
        feddaa_assignment = {ci: 0 for ci in range(n)}
        feddaa_event = "none"
        feddaa_split_score = 0.0
        feddaa_shift_clients = set()
        feddaa_clean_clients = set(range(n))
        feddaa_cluster_source = "none"
        feddaa_silhouette = 0.0
        feddaa_last_rebuild_round = None
        feddaa_previous_prototypes = {}
        feddaa_report_clients = 0
        feddaa_report_bytes = 0
        feddaa_report_round = "n/a"
        feddaa_model_evaluations = 0
        feddaa_assignment_changed_clients = 0
        feddaa_rebuild_count = 0
        feddaa_history_mix_clients = 0
        feddaa_sample_weight_entropy = 0.0
        spawn_models = None     # cid -> model, set when spawn_concept fires
        spawn_assignment = None
        client_runtime = [ClientRuntimeState() for _ in range(n)]
        fdms_M = np.eye(n, dtype=float)
        fdms_T = np.zeros((n, n), dtype=float)
        fdms_ready = False
        last_features = {}
        feddrift_confirm = np.zeros(n, dtype=int)
        feddrift_scores = np.zeros(n, dtype=float)
        feddrift_ref_acc = np.full(n, np.nan, dtype=float)
        feddrift_mark_until = {}
        feddrift_model_weight = {}
        feddrift_model_versions = {}
        feddrift_sc_weights = {}
        feddrift_assignment_log = []
        feddrift_next_cluster_id = 1
        feddrift_split_score = 0.0
        feddrift_merge_count = 0
        feddrift_split_count = 0
        feddrift_reused_clients = 0
        feddrift_pending_groups = []
        feddrift_first_trigger_round = None
        feddrift_assignment_changed_clients = 0
        feddrift_prev_loss_mean = None
        feddrift_loss_jump = 0.0
        previous_monitor_roster = set()
        previous_score_exceeders = set()
        cluster_purity = "n/a"
        hist = {"acc": [], "worst_acc": [], "lr": [], "switch_round": None,
                "tool": "none", "tool_bundle": [], "actions": [], "telemetry": [],
                "weight_entropy": [], "max_weight": [], "reweight_source": [],
                "robust_variant": [], "compromised_num": [], "robust_keep_n": []}
        if b3_plan:
            hist["hidden_event_log"] = [{
                "episode_id": b3_plan["episode_id"],
                "plan_sha256": b3_plan["plan_sha256"],
                "partition_sha256": b3_plan.get("partition_sha256"),
                "event_round": event_round,
                "active_causes": list(b3_plan["hidden_event"]["active_causes"]),
                "affected_clients": b3_plan["hidden_event"]["affected_clients"],
                "cause_intervals": b3_plan["hidden_event"].get("cause_intervals", {}),
                "event_schedule": b3_plan["hidden_event"].get("event_schedule", []),
            }]
        selected_override = None

        def record_tool(action_type, round_idx):
            if action_type not in hist["tool_bundle"]:
                hist["tool_bundle"].append(action_type)
            if hist["switch_round"] is None:
                hist["switch_round"] = round_idx
                hist["tool"] = action_type

        def feddrift_labels(ci, rr):
            if ci in concept_of and concept_of[ci] in drift_maps:
                return (drift_maps[concept_of[ci]][cy[ci]]
                        if rr >= schedule.get(ci, 10**9) else cy[ci])
            return episode_data(ci, rr)[1]

        def feddrift_client_acc(m, ci, rr):
            return _model_acc(m, cx[ci], feddrift_labels(ci, rr), device,
                              int(getattr(cfg, "feddrift_eval_samples", 0)))

        def feddrift_client_report(models, versions, ci, rr):
            return FedDriftReport(
                ci, rr, len(cy[ci]),
                tuple((mid, versions[mid], feddrift_client_acc(models[mid], ci, rr))
                      for mid in sorted(models)),
            )

        def clone_spawn_model(src):
            m = build_model(self.data["n_classes"], self.data["input_shape"]).to(device)
            m.load_state_dict(src.state_dict())
            return m

        def record_feddrift_sc_weights(rr, clients):
            if not spawn_models or not spawn_assignment:
                return
            decay = min(1.0, max(0.0, float(getattr(cfg, "feddrift_history_decay", 0.9))))
            cap = max(0.0, float(getattr(cfg, "feddrift_history_cap", 5.0)))
            for key in list(feddrift_sc_weights):
                feddrift_sc_weights[key] *= decay
                if feddrift_sc_weights[key] < 1e-6:
                    feddrift_sc_weights.pop(key)
            for ci in clients:
                cid = spawn_assignment.get(ci)
                if cid in spawn_models:
                    key = (cid, ci)
                    feddrift_sc_weights[key] = min(
                        cap, feddrift_sc_weights.get(key, 0.0) + 1.0
                    )
                    feddrift_assignment_log.append((rr, cid, ci, feddrift_model_versions[cid]))

        def feddrift_source_step(rr, reports, clients):
            nonlocal spawn_models, spawn_assignment, feddrift_next_cluster_id
            nonlocal feddrift_split_score, cluster_purity, feddrift_merge_count
            nonlocal feddrift_assignment_changed_clients, feddrift_split_count
            nonlocal feddrift_reused_clients
            if not spawn_models:
                return
            clients = tuple(clients)
            if not clients:
                return
            h_delta = float(getattr(cfg, "feddrift_h_delta", cfg.feddrift_loss_delta))
            h_deltap = float(getattr(cfg, "feddrift_h_deltap", h_delta))
            max_models = max(1, int(cfg.feddrift_max_concepts))
            mark_rounds = max(1, int(getattr(cfg, "feddrift_mark_rounds", 1)))
            spawn_assignment = dict(spawn_assignment or {ci: next(iter(spawn_models)) for ci in range(n)})
            previous_assignment = dict(spawn_assignment)
            protected = {mid for mid, until in feddrift_mark_until.values() if rr < until}
            expected = {(mid, feddrift_model_versions[mid]) for mid in spawn_models}
            if any(report.source_round != rr or report.client_id not in clients or
                   {(mid, version) for mid, version, _ in report.model_scores} != expected
                   for report in reports):
                raise ValueError("FedDrift report version/round mismatch")
            report_by_client = {report.client_id: report for report in reports}
            if set(report_by_client) != set(clients):
                raise ValueError("FedDrift reports must cover current online clients")
            confirmed, feddrift_split_score = _feddrift_update_reports(
                reports, feddrift_ref_acc, feddrift_confirm, feddrift_scores, cfg
            )
            acc_matrix = {mid: {ci: report_by_client[ci].scores()[mid] for ci in clients}
                          for mid in sorted(spawn_models)}
            new_assignment = dict(spawn_assignment)
            best_by_client = _feddrift_match_models(reports)
            feddrift_reused_clients = 0
            for ci in clients:
                marked = feddrift_mark_until.get(ci)
                if marked and rr < marked[1] and marked[0] in spawn_models:
                    new_assignment[ci] = marked[0]
                    continue
                if marked and rr >= marked[1]:
                    feddrift_mark_until.pop(ci, None)
                best_mid = best_by_client[ci]
                if previous_assignment.get(ci) != best_mid and feddrift_scores[ci] <= float(cfg.feddrift_h_delta):
                    feddrift_reused_clients += 1
                new_assignment[ci] = best_mid

            stable = _feddrift_stable_groups(confirmed, last_features, cfg)
            if stable and len(spawn_models) < max_models:
                group = [ci for ci in stable[0] if ci not in feddrift_mark_until]
                if len(group) >= int(cfg.feddrift_min_cluster_size):
                    source = max(set(best_by_client[ci] for ci in group),
                                 key=lambda mid: sum(best_by_client[ci] == mid for ci in group))
                    new_mid = feddrift_next_cluster_id
                    while new_mid in spawn_models:
                        new_mid += 1
                    feddrift_next_cluster_id = new_mid + 1
                    spawn_models[new_mid] = clone_spawn_model(spawn_models[source])
                    feddrift_model_versions[new_mid] = feddrift_model_versions[source]
                    feddrift_model_weight[new_mid] = 0.0
                    acc_matrix[new_mid] = dict(acc_matrix[source])
                    for ci in group:
                        feddrift_mark_until[ci] = (new_mid, rr + mark_rounds)
                        new_assignment[ci] = new_mid
                        feddrift_confirm[ci] = 0
                        feddrift_ref_acc[ci] = acc_matrix[source][ci]
                    protected.add(new_mid)
                    feddrift_split_count += 1

            spawn_assignment = new_assignment
            groups = {mid: [ci for ci in clients if spawn_assignment.get(ci) == mid]
                      for mid in sorted(spawn_models)}
            active = [mid for mid, members in groups.items() if members]
            feddrift_merge_count = 0
            while len(active) > 1:
                best = None
                for i, a in enumerate(active):
                    for b in active[i + 1:]:
                        if a in protected or b in protected:
                            continue
                        a_self = np.mean([acc_matrix[a][ci] for ci in groups[a]]) if groups[a] else 0.0
                        b_self = np.mean([acc_matrix[b][ci] for ci in groups[b]]) if groups[b] else 0.0
                        a_on_b = np.mean([acc_matrix[a][ci] for ci in groups[b]]) if groups[b] else 0.0
                        b_on_a = np.mean([acc_matrix[b][ci] for ci in groups[a]]) if groups[a] else 0.0
                        dist = max(float(a_self - a_on_b), float(b_self - b_on_a), 0.0)
                        if best is None or dist < best[0]:
                            best = (dist, a, b)
                if best is None or best[0] > h_deltap:
                    break
                _, a, b = best
                wa = feddrift_model_weight.get(a, len(groups.get(a, [])))
                wb = feddrift_model_weight.get(b, len(groups.get(b, [])))
                _merge_model_weights(spawn_models[a], spawn_models[b], wa, wb)
                feddrift_model_versions[a] += 1
                feddrift_model_weight[a] = float(wa) + float(wb)
                spawn_assignment = {ci: (a if lab == b else lab) for ci, lab in spawn_assignment.items()}
                spawn_models.pop(b, None)
                feddrift_model_weight.pop(b, None)
                feddrift_model_versions.pop(b, None)
                for key in list(feddrift_sc_weights):
                    mid, ci = key
                    if mid == b:
                        feddrift_sc_weights[(a, ci)] = (
                            feddrift_sc_weights.get((a, ci), 0.0) +
                            feddrift_sc_weights.pop(key))
                feddrift_mark_until.update({ci: (a, until) for ci, (mid, until) in feddrift_mark_until.items()
                                            if mid == b})
                groups = {mid: [ci for ci in clients if spawn_assignment.get(ci) == mid]
                          for mid in sorted(spawn_models)}
                active = [mid for mid, members in groups.items() if members]
                feddrift_merge_count += 1

            active_set = set(spawn_assignment.values())
            for mid in list(spawn_models):
                if mid not in active_set and mid not in protected:
                    spawn_models.pop(mid, None)
                    feddrift_model_weight.pop(mid, None)
                    feddrift_model_versions.pop(mid, None)
            for key in list(feddrift_sc_weights):
                if key[0] not in spawn_models:
                    feddrift_sc_weights.pop(key)
            feddrift_assignment_changed_clients = sum(
                previous_assignment.get(ci) != spawn_assignment.get(ci) for ci in clients)
            current_truth = {ci: concept_of[ci] if rr >= schedule.get(ci, 10**9) else 0
                             for ci in concept_of}
            cluster_purity = clustering_metrics(spawn_assignment, current_truth)["purity"]

        def episode_data(ci, rr, before=False):
            xc, yc = cx[ci], cy[ci]
            shifted = False
            if (not before and data_cause and
                    self._timeline().active(data_cause, rr, ci)):
                shifted = True
                if data_cause == "virtual_drift":
                    xc = self.data["vshift_X"][ci]
                elif data_cause == "label_prior_drift":
                    xc, yc = self.data["label_X"][ci], self.data["label_y"][ci]
                else:
                    yc = drift_map[yc]
            faulted = (not before and self._timeline().active("fault", rr, ci))
            return xc, yc, shifted, faulted

        def feddaa_data(ci, rr, before=False):
            xc, yc, _, _ = episode_data(ci, rr, before)
            return xc, yc

        def label_histogram_report(ci, rr, labels):
            counts = torch.bincount(labels.cpu(), minlength=self.data["n_classes"])
            return LabelHistogramReport(ci, rr, tuple(int(value) for value in counts))

        def feddaa_client_report(models, ci, rr):
            xc, yc = feddaa_data(ci, rr)
            counts = torch.bincount(yc.cpu(), minlength=self.data["n_classes"])
            prototype = _model_proto(
                models[0], xc, yc, self.data["n_classes"], device
            )
            return FedDAAReport(
                ci, rr, len(yc), self.data["n_classes"],
                tuple(float(value) for value in prototype.ravel()),
                tuple(int(value) for value in counts),
                tuple((cid, _model_loss(candidate, xc, yc, device))
                      for cid, candidate in enumerate(models)),
            )

        def add_feddaa_substitute(ci, state, weight):
            if not feddaa_enabled or not feddaa_cluster_updates:
                return
            cid = int(np.argmax(feddaa_weights[ci, :len(feddaa_models)]))
            feddaa_cluster_updates[cid][0].append(state)
            feddaa_cluster_updates[cid][1].append(weight)
            feddaa_cluster_updates[cid][2].append(ci)

        def feddaa_rebuild(rr, reason, reports):
            nonlocal feddaa_models, feddaa_weights, feddaa_assignment, feddaa_label_history
            nonlocal feddaa_event, feddaa_split_score, feddaa_shift_clients, feddaa_clean_clients
            nonlocal feddaa_cluster_source, feddaa_silhouette, feddaa_last_rebuild_round
            nonlocal feddaa_previous_prototypes
            nonlocal feddaa_assignment_changed_clients, feddaa_rebuild_count
            if not feddaa_models:
                return
            reports = tuple(reports)
            clients = tuple(report.client_id for report in reports)
            if not reports:
                return
            if (len(clients) != len(set(clients)) or
                    any(report.source_round != rr or
                        set(report.losses()) != set(range(len(feddaa_models)))
                        for report in reports)):
                raise ValueError("FedDAA report round/model mismatch")
            current_proto = {
                report.client_id: np.asarray(report.prototype).reshape(
                    report.n_classes, report.n_classes
                ).ravel()
                for report in reports
            }
            new_assignment, feddaa_silhouette, _ = _feddaa_auto_assignment(
                current_proto, cfg, cfg.seed + 9000 + rr)
            feddaa_shift_clients = {
                ci for ci in clients if ci in feddaa_previous_prototypes and
                float(np.linalg.norm(current_proto[ci] - feddaa_previous_prototypes[ci])) >
                float(cfg.feddaa_tol_merge)
            }
            feddaa_clean_clients = set(clients) - feddaa_shift_clients
            feddaa_previous_prototypes.update(current_proto)

            old_models = feddaa_models
            old_assignment = dict(feddaa_assignment)
            old_history = feddaa_label_history
            new_k = max(new_assignment.values(), default=0) + 1
            new_models, new_history = [], []
            for cid in range(new_k):
                members = [ci for ci, lab in new_assignment.items() if lab == cid]
                counts = {}
                for ci in members:
                    old = min(old_assignment.get(ci, 0), len(old_models) - 1)
                    counts[old] = counts.get(old, 0) + 1
                src = max(counts, key=counts.get) if counts else 0
                m = build_model(self.data["n_classes"], self.data["input_shape"]).to(device)
                m.load_state_dict(old_models[src].state_dict())
                new_models.append(m)
                new_history.append(old_history[src])
            feddaa_models = new_models
            feddaa_label_history = np.asarray(new_history, dtype=float)
            feddaa_assignment = {
                ci: new_assignment.get(ci, min(old_assignment.get(ci, 0), new_k - 1))
                for ci in range(n)
            }
            feddaa_assignment_changed_clients = (0 if reason == "init" else sum(
                old_assignment.get(ci) != feddaa_assignment.get(ci) for ci in clients
            ))
            feddaa_weights = np.zeros((n, new_k), dtype=float)
            for ci, cid in feddaa_assignment.items():
                feddaa_weights[ci, cid] = 1.0
            feddaa_split_score = feddaa_silhouette
            feddaa_event = reason
            feddaa_cluster_source = "client_conditional_prototype_silhouette"
            feddaa_last_rebuild_round = rr
            feddaa_rebuild_count += 1

        for r in range(cfg.rounds):
            self.current_round = r
            applied_override = selected_override
            present = self.present_clients(r, applied_override)
            if b3_plan:
                available_now = set(getattr(self, "last_available_clients", present))
                planned_now = getattr(self, "last_planned_clients", present)
            else:
                available_now = {
                    ci for ci in range(n)
                    if not self._timeline().active("dropout", r, ci)
                }
                planned_now = present
            for ci, runtime in enumerate(client_runtime):
                runtime.observe_availability(
                    ci in available_now, r, selected=ci in planned_now
                )
            selected_override = None
            select_invalid = getattr(self, "last_select_invalid", 0)
            select_used = getattr(self, "last_select_used", False)
            c_losses, u_norms, client_reports = [], [], []
            dropout_handle_count = 0
            friend_count, friend_sims, friend_errors, friend_candidate_counts = 0, [], [], []
            round_client_weights = None
            reweight_source = "sample_count"
            label_prior_source = "none"
            label_prior_report_clients = 0
            moment_align_clients = 0
            feddrift_assignment_changed_clients = 0
            feddrift_reused_clients = 0
            feddrift_loss_jump = 0.0
            feddrift_pending_groups = []
            feddaa_event = "none"
            feddaa_assignment_changed_clients = 0
            feddaa_history_mix_clients = 0
            feddaa_sample_weight_entropy = 0.0
            data_shift_clients, fault_injected_clients = set(), set()
            final_norms, aggregation_sources = [], []
            robust_info = {"variant": "none", "compromised_num": 0,
                           "keep_n": 0, "selected": [], "trigger_source": "none"}
            online_label_prior = None
            if (data_cause == "label_prior_drift" and present and
                    (use_label_prior_adapt or use_reweight)):
                label_reports = [
                    label_histogram_report(ci, r, episode_data(ci, r)[1])
                    for ci in present
                ]
                online_label_prior = _aggregate_label_reports(
                    label_reports, self.data["n_classes"], r
                )
                label_prior_report_clients = len(label_reports)

            if staggered and use_spawn:
                # multi-model: each client trains the model of its concept cluster;
                # auto routing uses learned assignment; oracle routing uses truth.
                if spawn_auto:
                    feddrift_reports = [
                        feddrift_client_report(
                            spawn_models, feddrift_model_versions, ci, r
                        ) for ci in present
                    ]
                    feddrift_source_step(r, feddrift_reports, present)
                cluster = {cid: ([], []) for cid in spawn_models}   # cid -> (states, weights)
                for ci in present:
                    cid = spawn_assignment.get(ci, 1) if spawn_assignment else concept_of[ci]
                    if cid not in spawn_models:
                        cid = next(iter(spawn_models))
                    gstate_c = {k: v.detach().clone() for k, v in spawn_models[cid].state_dict().items()}
                    true_cid = concept_of[ci]
                    yc = drift_maps[true_cid][cy[ci]] if r >= schedule.get(ci, 10**9) else cy[ci]
                    st, un, loss, report = self._train_client(
                        local, gstate_c, cx[ci], yc, current_lr,
                        baseline_inputs=cx[ci],
                        baseline_labels=cy[ci],
                    )
                    last_features[ci] = _update_sketch(st, gstate_c)
                    sc_weight = 1.0
                    if spawn_auto:
                        # ponytail: source sc_weights as aggregation weight;
                        # full historical replay belongs in a FedML runner clone.
                        sc_weight += feddrift_sc_weights.get((cid, ci), 0.0)
                    cluster[cid][0].append(st); cluster[cid][1].append(len(yc) * sc_weight)
                    c_losses.append(loss); u_norms.append(un)
                    client_reports.append(report)
                    client_last_loss[ci] = loss; client_seen_rounds[ci] += 1
                for cid, (sts, wts) in cluster.items():
                    if sts:
                        spawn_models[cid].to("cpu"); aggregate(spawn_models[cid], sts, wts)
                        spawn_models[cid].to(device)
                        feddrift_model_versions[cid] += 1
                        feddrift_model_weight[cid] = feddrift_model_weight.get(cid, 0.0) + float(sum(wts))
                if spawn_auto:
                    record_feddrift_sc_weights(r, present)
                    cluster_purity = _feddrift_cluster_report(spawn_assignment, concept_of)
                final_norms = list(u_norms)
                aggregation_sources = ([{"source": "online_real", "count": len(u_norms)}]
                                       if u_norms else [])
            else:
                gstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if feddaa_enabled and not feddaa_models:
                    m = build_model(self.data["n_classes"], self.data["input_shape"]).to(device)
                    m.load_state_dict(model.state_dict())
                    feddaa_models = [m]
                feddaa_reports = ([feddaa_client_report(feddaa_models, ci, r) for ci in present]
                                  if feddaa_enabled and present else [])
                feddaa_report_clients = len(feddaa_reports)
                feddaa_report_bytes = sum(report.communication_bytes for report in feddaa_reports)
                feddaa_report_round = r if feddaa_reports else "n/a"
                feddaa_model_evaluations = sum(len(report.model_losses)
                                               for report in feddaa_reports)
                rebuilt = False
                if feddaa_reports and feddaa_last_rebuild_round is None:
                    prior = _aggregate_label_reports(
                        [LabelHistogramReport(report.client_id, report.source_round,
                                              report.class_counts)
                         for report in feddaa_reports],
                        self.data["n_classes"], r,
                    )
                    feddaa_label_history = np.asarray([prior])
                    feddaa_rebuild(r, "init", feddaa_reports)
                    rebuilt = True
                elif (feddaa_reports and
                      _feddaa_rebuild_due(r, feddaa_last_rebuild_round, cfg.feddaa_T)):
                    feddaa_rebuild(r, "prototype_silhouette", feddaa_reports)
                    rebuilt = True
                if feddaa_reports and not rebuilt:
                    report_by_client = {report.client_id: report for report in feddaa_reports}
                    for ci, report in report_by_client.items():
                        losses = [report.losses()[cid] for cid in range(len(feddaa_models))]
                        if use_reweight and feddaa_mode == "label":
                            hist = np.asarray(report.class_counts, dtype=float) / report.sample_count
                            logits = [-(loss / max(float(cfg.feddaa_soft_temp), 1e-6)) +
                                      float(cfg.feddaa_label_weight) *
                                      float(np.dot(hist, -np.log(np.maximum(feddaa_label_history[cid], 1e-6))))
                                      for cid, loss in enumerate(losses)]
                            feddaa_weights[ci, :len(feddaa_models)] = _softmax(logits)
                        else:
                            feddaa_weights[ci, :len(feddaa_models)] = _softmax_neg(
                                losses, cfg.feddaa_soft_temp
                            )
                    feddaa_label_history = _update_label_history(
                        feddaa_label_history, feddaa_weights, report_by_client,
                        float(cfg.feddaa_history_momentum))
                    ww = feddaa_weights[list(report_by_client), :len(feddaa_models)].mean(axis=0)
                    round_client_weights = (ww / max(float(ww.sum()), 1e-12)).tolist()
                    ww_norm = ww / max(float(ww.sum()), 1e-12)
                    feddaa_sample_weight_entropy = float(
                        -(ww_norm * np.log(ww_norm + 1e-12)).sum()
                    )
                feddaa_cluster_updates = {i: ([], [], []) for i in range(len(feddaa_models))}
                states, weights, state_client_ids, states_by_client = [], [], [], {}
                update_envelopes = []
                back = bool(data_cause and r >= event_round and
                            not self._timeline().active(data_cause, r))
                for ci in present:
                    xc, yc, shifted, faulted = episode_data(ci, r)
                    if shifted:
                        data_shift_clients.add(ci)
                    if faulted:
                        fault_injected_clients.add(ci)
                    class_weight = None
                    logit_calibration = None
                    train_gstate = gstate
                    feddaa_cid = 0
                    if feddaa_enabled:
                        if (feddaa_mode == "virtual" and ci in feddaa_clean_clients and
                                self._timeline().active(data_cause, r, ci)):
                            x0, y0 = feddaa_data(ci, r, before=True)
                            # ponytail: source merges old+current dataloaders; keep
                            # one client-sized mixed batch so validation remains usable.
                            keep = max(len(y0), len(yc))
                            old_n = keep // 2
                            cur_n = keep - old_n
                            xc, yc = torch.cat([x0[:old_n], xc[:cur_n]]), torch.cat([y0[:old_n], yc[:cur_n]])
                            feddaa_history_mix_clients += 1
                        class_weight = self.data["label_class_weight"].to(device) if feddaa_mode == "label" else None
                        best_st, best_un, best_loss, best_report, best_cid = (
                            None, 0.0, float("inf"), None, 0
                        )
                        loss_sum, weight_sum = 0.0, 0.0
                        for cid, m_c in enumerate(feddaa_models):
                            lw = float(feddaa_weights[ci, cid]) if cid < feddaa_weights.shape[1] else 0.0
                            if lw <= 1e-6:
                                continue
                            train_gstate_c = {k: v.detach().clone() for k, v in m_c.state_dict().items()}
                            st_c, un_c, loss_c, report_c = self._train_client(
                                local, train_gstate_c, xc, yc, current_lr,
                                prox_mu=cfg.fedprox_mu if use_fedprox else 0.0,
                                class_weight=class_weight,
                                baseline_inputs=cx[ci],
                                baseline_labels=cy[ci],
                                byzantine_scale=(cfg.byzantine_scale if faulted else 0.0))
                            feddaa_cluster_updates[cid][0].append(st_c)
                            feddaa_cluster_updates[cid][1].append(len(yc) * lw)
                            feddaa_cluster_updates[cid][2].append(ci)
                            loss_sum += lw * loss_c
                            weight_sum += lw
                            if loss_c < best_loss:
                                best_st, best_un, best_loss, best_report, best_cid = (
                                    st_c, un_c, loss_c, report_c, cid
                                )
                        if best_st is None:
                            best_cid = int(np.argmax(feddaa_weights[ci, :len(feddaa_models)]))
                            train_gstate_c = {k: v.detach().clone() for k, v in feddaa_models[best_cid].state_dict().items()}
                            best_st, best_un, best_loss, best_report = self._train_client(
                                local, train_gstate_c, xc, yc, current_lr,
                                prox_mu=cfg.fedprox_mu if use_fedprox else 0.0,
                                class_weight=class_weight,
                                baseline_inputs=cx[ci],
                                baseline_labels=cy[ci],
                                byzantine_scale=(cfg.byzantine_scale if faulted else 0.0))
                            feddaa_cluster_updates[best_cid][0].append(best_st)
                            feddaa_cluster_updates[best_cid][1].append(len(yc))
                            feddaa_cluster_updates[best_cid][2].append(ci)
                            weight_sum = 1.0; loss_sum = best_loss
                        st, un, loss, report = (
                            best_st, best_un, float(loss_sum / max(weight_sum, 1e-12)),
                            best_report,
                        )
                        feddaa_cid = best_cid
                        train_gstate = {k: v.detach().clone() for k, v in feddaa_models[feddaa_cid].state_dict().items()}
                    elif (shifted or (staggered and r >= schedule.get(ci, 10**9))) and not back:
                        if staggered:
                            yc = drift_maps[concept_of[ci]][yc]
                        elif data_cause == "virtual_drift":
                            if use_moment_align:
                                xc = self._moment_align(xc)
                                moment_align_clients += 1
                        elif data_cause == "label_prior_drift":
                            if use_label_prior_adapt and not feddaa_enabled:
                                logit_calibration = torch.as_tensor(
                                    -np.log(np.maximum(online_label_prior, 1e-6)),
                                    dtype=torch.float32, device=device,
                                )
                                label_prior_source = "online_client_histogram"
                            elif use_reweight and not feddaa_enabled:
                                logit_calibration = torch.as_tensor(
                                    float(cfg.fedlc_tau) *
                                    np.maximum(online_label_prior, 1e-6) ** -0.25,
                                    dtype=torch.float32, device=device,
                                )
                                reweight_source = "fedlc_logit_calibration"
                    if not feddaa_enabled:
                        st, un, loss, report = self._train_client(
                            local, train_gstate, xc, yc, current_lr,
                            prox_mu=cfg.fedprox_mu if use_fedprox else 0.0,
                            class_weight=class_weight,
                            logit_calibration=logit_calibration,
                            baseline_inputs=cx[ci],
                            baseline_labels=cy[ci],
                            byzantine_scale=(cfg.byzantine_scale if faulted else 0.0),
                        )
                    states_by_client[ci] = (st, len(yc))
                    last_features[ci] = _update_sketch(st, train_gstate)
                    states.append(st); weights.append(len(yc)); state_client_ids.append(ci)
                    update_envelopes.append(UpdateEnvelope(
                        ci, r, "online_real", ci, "unverified", st, len(yc)
                    ))
                    c_losses.append(loss); u_norms.append(un)
                    client_reports.append(report)
                    client_last_loss[ci] = loss; client_seen_rounds[ci] += 1
                    _oort_update_client(oort_state, ci, loss * np.sqrt(max(1.0, client_sizes[ci])), r)

                fdms_sims = _fdms_pairwise_similarity(states_by_client, gstate) if states_by_client else {}
                # dropout_handle: substitute absent clients with their last-seen update
                if present and use_dropout_handle:
                    for ci in range(n):
                        cached_update = client_runtime[ci].usable_cache(
                            r, int(cfg.dropout_cache_ttl)
                        )
                        if ci not in present and cached_update is not None:
                            st_c, w_c = cached_update.state_dict, cached_update.sample_count
                            states.append(st_c); weights.append(w_c); state_client_ids.append(ci)
                            update_envelopes.append(UpdateEnvelope(
                                ci, cached_update.source_round, "stale_cache",
                                cached_update.origin_client, "trusted", st_c, w_c,
                            ))
                            add_feddaa_substitute(ci, st_c, w_c)
                            dropout_handle_count += 1
                if use_friend_substitute:
                    for ci in range(n):
                        if ci in present:
                            continue
                        fj, sim, candidate_count = self.friend_for(ci, present, fdms_M, fdms_ready)
                        friend_candidate_counts.append(candidate_count)
                        if fj is not None and fj in states_by_client:
                            st_f, _ = states_by_client[fj]
                            states.append(st_f); weights.append(len(cy[ci])); state_client_ids.append(ci)
                            update_envelopes.append(UpdateEnvelope(
                                ci, r, "friend_substitute", fj, "current_online",
                                st_f, len(cy[ci]),
                            ))
                            add_feddaa_substitute(ci, st_f, len(cy[ci]))
                            friend_count += 1; friend_sims.append(sim)
                if fdms_sims:
                    _fdms_update_matrix(fdms_M, fdms_T, fdms_sims)
                    fdms_ready = True

                model.to("cpu")
                accepted_real_clients = set()
                final_indices = []
                if not states:
                    robust_info = {"variant": "none", "compromised_num": 0,
                                   "keep_n": 0, "selected": [],
                                   "trigger_source": "empty_round"}
                elif feddaa_enabled:
                    selected_ids = None
                    if use_robust:
                        requested = int(float(cfg.byzantine_frac) * len(states))
                        if cfg.robust_variant == "coord_trimmed_mean":
                            final_indices = list(range(len(states)))
                            actual = max(0, min(int(len(states) * cfg.robust_trim),
                                                (len(states) - 1) // 2))
                            robust_info = {"variant": "coord_trimmed_mean",
                                           "compromised_num": actual,
                                           "keep_n": max(1, len(states) - 2 * actual),
                                           "selected": state_client_ids,
                                           "trigger_source": "feddaa_cluster_coord_trim"}
                        else:
                            chosen, actual = _distance_trimmed_indices(states, requested)
                            final_indices = chosen
                            selected_ids = {state_client_ids[i] for i in chosen}
                            accepted_real_clients = {
                                update_envelopes[i].client_id for i in chosen
                                if update_envelopes[i].source_kind == "online_real"
                            }
                            robust_info = {"variant": "distance_trimmed_mean",
                                           "compromised_num": actual,
                                           "keep_n": len(chosen),
                                           "selected": [state_client_ids[i] for i in chosen],
                                           "trigger_source": "feddaa_cluster_update_norm_outlier"}
                    for cid, (sts, wts, cids) in feddaa_cluster_updates.items():
                        if selected_ids is not None:
                            keep = [i for i, ci in enumerate(cids) if ci in selected_ids]
                            sts, wts = [sts[i] for i in keep], [wts[i] for i in keep]
                        if sts:
                            feddaa_models[cid].to("cpu")
                            if use_robust and cfg.robust_variant == "coord_trimmed_mean":
                                aggregate_robust(feddaa_models[cid], sts, cfg.robust_trim)
                            else:
                                aggregate(feddaa_models[cid], sts, wts)
                            feddaa_models[cid].to(device)
                    cluster_weights = np.zeros(len(feddaa_models), dtype=float)
                    for ci in range(n):
                        for cid in range(len(feddaa_models)):
                            cluster_weights[cid] += float(feddaa_weights[ci, cid]) * len(cy[ci])
                    cluster_weights = np.maximum(cluster_weights, 1.0)
                    for m in feddaa_models:
                        m.to("cpu")
                    aggregate(model, [{k: v.detach().cpu() for k, v in m.state_dict().items()}
                                      for m in feddaa_models], cluster_weights.tolist())
                    for m in feddaa_models:
                        m.to(device)
                    reweight_source = "feddaa_label_history_weights" if use_reweight else "feddaa_cluster_weights"
                    if not use_robust:
                        final_indices = list(range(len(states)))
                        accepted_real_clients = {
                            update.client_id for update in update_envelopes
                            if update.source_kind == "online_real"
                        }
                if not states:
                    pass
                elif feddaa_enabled and not use_robust:
                    robust_info = {"variant": "none", "compromised_num": 0,
                                   "keep_n": len(present), "selected": present,
                                   "trigger_source": "feddaa_cluster"}
                elif use_robust:
                    requested = int(float(cfg.byzantine_frac) * len(states))
                    if cfg.robust_variant == "coord_trimmed_mean":
                        final_indices = list(range(len(states)))
                        aggregate_robust(model, states, cfg.robust_trim)
                        actual = max(0, min(int(len(states) * cfg.robust_trim),
                                            (len(states) - 1) // 2))
                        robust_info = {"variant": "coord_trimmed_mean",
                                       "compromised_num": actual,
                                       "keep_n": max(1, len(states) - 2 * actual),
                                       "selected": state_client_ids,
                                       "trigger_source": "update_norm_outlier"}
                    else:
                        info = aggregate_distance_trimmed_mean(model, states, weights, requested)
                        final_indices = info["selected"]
                        accepted_real_clients = {
                            update_envelopes[i].client_id for i in info["selected"]
                            if update_envelopes[i].source_kind == "online_real"
                        }
                        robust_info = {"variant": "distance_trimmed_mean",
                                       "compromised_num": info["compromised_num"],
                                       "keep_n": info["keep_n"],
                                       "selected": [state_client_ids[i] for i in info["selected"]],
                                       "trigger_source": "update_norm_outlier"}
                else:
                    aggregate(model, states, weights if round_client_weights is None else round_client_weights)
                    final_indices = list(range(len(states)))
                    accepted_real_clients = {
                        update.client_id for update in update_envelopes
                        if update.source_kind == "online_real"
                    }
                for update in update_envelopes:
                    if update.source_kind != "online_real":
                        continue
                    runtime = client_runtime[update.client_id]
                    if update.client_id in accepted_real_clients:
                        runtime.accept(update)
                    elif use_robust:
                        runtime.reject("robust_rejected_or_unverifiable")
                final_norms = [
                    float(torch.linalg.vector_norm(_flat_update(
                        update_envelopes[index].state_dict, gstate
                    )))
                    for index in final_indices
                ]
                source_counts = {}
                for index in final_indices:
                    source = update_envelopes[index].source_kind
                    source_counts[source] = source_counts.get(source, 0) + 1
                aggregation_sources = [
                    {"source": source, "count": source_counts[source]}
                    for source in sorted(source_counts)
                ]
                model.to(device)

            if staggered:
                acc, worst = self.evaluate_staggered(None if use_spawn else model,
                                                     spawn_models if use_spawn else None,
                                                     r,
                                                     auto=spawn_auto,
                                                     assignment=spawn_assignment)
            else:
                acc, worst = self.evaluate(model, r, moment_align=use_moment_align)
            hist["acc"].append(acc); hist["worst_acc"].append(worst); hist["lr"].append(current_lr)
            if not use_spawn:
                reports = [feddrift_client_report({0: model}, {0: r}, ci, r)
                           for ci in present]
                confirmed, feddrift_split_score = _feddrift_update_reports(
                    reports, feddrift_ref_acc, feddrift_confirm, feddrift_scores, cfg
                )
                feddrift_pending_groups = _feddrift_stable_groups(
                    confirmed, last_features, cfg
                )
            loss_mean = float(np.mean(c_losses)) if c_losses else 0.0
            if feddrift_prev_loss_mean is not None:
                feddrift_loss_jump = loss_mean - feddrift_prev_loss_mean
            feddrift_prev_loss_mean = loss_mean
            if len(client_reports) != len(present):
                raise ValueError("client report roster mismatch")
            input_shift_reports = [
                InputShiftReport(
                    client_id=ci,
                    source_round=r,
                    baseline_sum=report["baseline_input_sum"],
                    baseline_count=report["baseline_input_count"],
                    current_sum=report["input_sum"],
                    current_count=report["input_count"],
                )
                for ci, report in zip(present, client_reports)
            ]
            in_shift = _simulated_secure_input_shift(input_shift_reports, r)
            shift_reports = [
                LabelShiftReport(
                    client_id=ci,
                    source_round=r,
                    baseline_counts=tuple(report["baseline_label_counts"]),
                    current_counts=tuple(report["label_counts"]),
                )
                for ci, report in zip(present, client_reports)
            ]
            lab_shift = _simulated_secure_label_shift(
                shift_reports, self.data["n_classes"], r
            )
            obs = collect(r, acc, hist["acc"], c_losses, u_norms, current_lr,
                          len(present) / n, use_robust, cfg.rounds,
                          input_shift=in_shift, label_shift=lab_shift)
            obs["decision_interval"] = int(cfg.decide_every)
            if b3_plan:
                planned = getattr(self, "last_planned_clients", present)
                available = getattr(self, "last_available_clients", present)
                available_planned = getattr(
                    self, "last_available_planned_clients", present
                )
                planned_rate = len(planned) / n
                planned_available_rate = len(available_planned) / n
                availability_rate = len(available) / n
            else:
                planned_rate = (max(1, int(cfg.participation * n)) / n
                                if cfg.participation < 1.0 else 1.0)
                availability_rate = len(present) / n
                planned_available_rate = availability_rate
            obs.update({
                "planned_participation_rate": round(float(planned_rate), 3),
                "availability_rate": round(float(availability_rate), 3),
                "participation_gap": round(
                    float(max(0.0, planned_rate - planned_available_rate)), 3
                ),
                "active_actions": list(action_state.active),
                "empty_round": not bool(present),
            })
            feddrift_drifted = [ci for ci in present
                                if feddrift_confirm[ci] >= int(cfg.feddrift_confirm_rounds)]
            feddrift_trigger = bool(not use_spawn and feddrift_pending_groups)
            if feddrift_trigger and feddrift_first_trigger_round is None:
                feddrift_first_trigger_round = r
            score_drops = np.maximum(
                0.0, feddrift_scores[np.asarray(present, dtype=int)]
            ) if present else np.asarray([], dtype=float)
            feature_clients = [ci for ci in present if ci in last_features]
            # ponytail: O(n^2) over the benchmark's <=20 clients; sample pairs if cohorts grow.
            direction_distances = [
                _cosine_distance(last_features[feature_clients[i]],
                                 last_features[feature_clients[j]])
                for i in range(len(feature_clients))
                for j in range(i + 1, len(feature_clients))
            ]
            current_roster = set(present)
            common_roster = current_roster & previous_monitor_roster
            roster_union = current_roster | previous_monitor_roster
            score_threshold = float(
                getattr(cfg, "feddrift_h_delta", cfg.feddrift_loss_delta)
            )
            score_exceeders = {
                ci for ci in present if feddrift_scores[ci] > score_threshold
            }
            common_score_union = (
                score_exceeders | previous_score_exceeders
            ) & common_roster
            high_features = [ci for ci in feature_clients if ci in score_exceeders]
            other_features = [ci for ci in feature_clients if ci not in score_exceeders]
            within_high = [
                _cosine_distance(last_features[high_features[i]],
                                 last_features[high_features[j]])
                for i in range(len(high_features))
                for j in range(i + 1, len(high_features))
            ]
            between_groups = [
                _cosine_distance(last_features[i], last_features[j])
                for i in high_features for j in other_features
            ]
            min_monitor_clients = max(
                6, 2 * int(cfg.feddrift_min_cluster_size)
            )
            obs.update({
                "client_change_monitor_ready": bool(
                    r >= int(cfg.feddrift_warmup_rounds) and
                    len(feature_clients) >= min_monitor_clients and
                    len(common_roster) >= min_monitor_clients
                ),
                "client_model_score_drop_p50": round(
                    float(np.quantile(score_drops, 0.5)), 6
                ) if score_drops.size else 0.0,
                "client_model_score_drop_p90": round(
                    float(np.quantile(score_drops, 0.9)), 6
                ) if score_drops.size else 0.0,
                "client_update_direction_dispersion": round(
                    float(np.mean(direction_distances)), 6
                ) if direction_distances else 0.0,
                "online_roster_overlap": round(
                    len(common_roster) / max(1, len(roster_union)), 6
                ),
                "client_score_exceedance_fraction": round(
                    len(score_exceeders) / max(1, len(current_roster)), 6
                ),
                "client_score_exceedance_overlap": round(
                    len(score_exceeders & previous_score_exceeders & common_roster) /
                    max(1, len(common_score_union)), 6
                ),
                "client_update_group_separation": round(
                    float(np.mean(between_groups) - np.mean(within_high)), 6
                ) if within_high and between_groups and len(other_features) >= 2 else 0.0,
                "client_loss_mean_delta": round(float(feddrift_loss_jump), 6),
            })
            previous_monitor_roster = current_roster
            previous_score_exceeders = score_exceeders
            current_truth = {
                ci: concept_of[ci] if r >= schedule.get(ci, 10**9) else 0
                for ci in concept_of
            }
            feddrift_structure = clustering_metrics(
                spawn_assignment or {}, current_truth
            )
            if feddrift_structure["purity"] != "n/a":
                cluster_purity = feddrift_structure["purity"]
            if present:
                oort_state["exploit_history"].append(float(np.mean([oort_state["reward"][ci] for ci in present])))
            _oort_pacer(oort_state, cfg)
            utilities, oort_prefer_duration = self.client_utilities(client_last_loss, client_seen_rounds,
                                                                    client_coverage, client_sizes, r,
                                                                    oort_state)
            reward_cap = float(np.median(oort_state["reward"]))
            for utility, runtime in zip(utilities, client_runtime):
                returned = runtime.returned_round == r
                utility.update({
                    "available": runtime.availability,
                    "runtime_state": runtime.return_state,
                    "selection_attempts": runtime.selection_attempts,
                    "last_success_round": runtime.last_success_round,
                    "cache_age": runtime.cache_age,
                    "rejection_reason": runtime.rejection_reason,
                    "oort_returned": returned,
                })
                if returned:
                    utility["oort_unexplored"] = True
                    utility["oort_reward"] = min(float(utility["oort_reward"]), reward_cap)
            oort_state["exploration"] = max(float(cfg.oort_exploration_min),
                                            float(oort_state["exploration"] * cfg.oort_exploration_decay))
            obs["client_utilities"] = utilities
            obs["select_k"] = max(1, int(cfg.participation * n)) if cfg.participation < 1.0 else n
            obs["select_enabled"] = bool(cfg.participation < 1.0)
            obs["select_seed"] = int(cfg.seed + 8000 + r)
            obs["oort_exploration"] = round(float(oort_state["exploration"]), 6)
            obs["oort_round_threshold"] = round(float(oort_state["round_threshold"]), 6)
            obs["oort_prefer_duration"] = round(float(oort_prefer_duration), 6)
            obs["oort_sample_window"] = int(cfg.oort_sample_window)
            final_stats = norm_stats(final_norms)
            obs.update({
                f"aggregation_norm_{key}": value for key, value in final_stats.items()
            })
            obs["aggregation_sources"] = aggregation_sources
            public_obs = public_observation(obs)
            weight_entropy = 0.0
            max_weight = 0.0
            if round_client_weights:
                ww = np.asarray(round_client_weights, dtype=float)
                weight_entropy = float(-(ww * np.log(ww + 1e-12)).sum())
                max_weight = float(ww.max())
            hist["weight_entropy"].append(weight_entropy)
            hist["max_weight"].append(max_weight)
            hist["reweight_source"].append(reweight_source if use_reweight else "sample_count")
            hist["robust_variant"].append(robust_info["variant"])
            hist["compromised_num"].append(robust_info["compromised_num"])
            hist["robust_keep_n"].append(robust_info["keep_n"])
            if hasattr(agent, "called_this_round"):
                agent.called_this_round = False
            decision_due = getattr(agent, "always_decide", False) or r % cfg.decide_every == 0
            raw_decision = agent.decide(public_obs) if decision_due else None
            decision_bundle_status = (getattr(agent, "last_bundle_status", "VALID")
                                      if decision_due else "VALID")
            if decision_bundle_status not in {
                "VALID", "INVALID_SCHEMA", "UNKNOWN_ACTION", "CONFLICT"
            }:
                decision_bundle_status = "INVALID_SCHEMA"
            bundle_invalid = decision_bundle_status != "VALID"
            if decision_due:
                try:
                    bundle = (ActionBundle(("no_op",)) if bundle_invalid
                              else as_action_bundle(raw_decision))
                except (TypeError, ValueError):
                    bundle = ActionBundle(("no_op",))
                    bundle_invalid = True
                    decision_bundle_status = "INVALID_SCHEMA"
            else:
                bundle = ActionBundle(("no_op",))
            round_actions = bundle.ordered_actions
            action = round_actions[0]  # B1/B2 compatibility field.
            action_types = tuple(item.type for item in round_actions)
            declared = getattr(agent, "last_declared_actions", None)
            declared_types = (tuple(declared) if decision_due and declared
                              else action_types if decision_due and not bundle_invalid else ())
            raw_output = getattr(agent, "last_raw_output", None) if decision_due else ""
            if decision_due and raw_output is None:
                raw_output = (json.dumps({"actions": list(declared_types)}, sort_keys=True)
                              if declared_types else repr(raw_decision))
            observation_json = (
                getattr(agent, "last_public_observation_json", "")
                if decision_due else ""
            ) or serialize_public_observation(public_obs)
            interface_error = (getattr(agent, "last_api_error", "") or
                               getattr(agent, "last_budget_error", "")) if decision_due else ""
            select_action = next((item for item in round_actions if item.type == "select_clients"), None)
            activated_actions = tuple(
                item for item in round_actions
                if item.type not in {"spawn_concept", "spawn_concept_auto"} or staggered
            )
            activated_bundle = ActionBundle(activated_actions or ("no_op",))
            transition = reduce_action_state(
                action_state, activated_bundle if decision_due else None
            )
            if decision_due and hasattr(agent, "record_decision_feedback"):
                agent.record_decision_feedback(
                    round_idx=r,
                    declared_actions=declared_types,
                    status="INTERFACE_ERROR" if interface_error else decision_bundle_status,
                    activated_actions=tuple(
                        name for name in activated_bundle.action_types if name != "no_op"
                    ),
                    started_actions=transition.started,
                    stopped_actions=transition.stopped,
                    effective_from_round=r + 1,
                    observation=public_obs,
                )
            hist["actions"].extend((r, item.type) for item in round_actions)
            b3_telemetry = (private_scorer_record({
                "episode_id": b3_plan["episode_id"],
                "planned_clients_count": len(getattr(self, "last_planned_clients", present)),
                "available_clients_count": len(getattr(self, "last_available_clients", present)),
                "participating_clients_count": len(present),
                "dropout_removed_clients": (n - len(getattr(self, "last_available_clients", present))),
                "data_shift_injected_clients": len(data_shift_clients),
                "fault_injected_clients": len(fault_injected_clients),
                "robust_filtered_clients": max(0, len(states) - int(robust_info["keep_n"]))
                                           if robust_info["variant"] != "none" else 0,
                "drifted_client_count": len(feddrift_drifted),
                "cluster_split_score": round(float(feddrift_split_score), 6),
                "feddrift_trigger": bool(feddrift_trigger),
                "feddrift_loss_delta": float(cfg.feddrift_loss_delta),
                "trusted_cache_count": sum(
                    runtime.trusted_cache is not None for runtime in client_runtime
                ),
                "cache_rejection_count": sum(
                    bool(runtime.rejection_reason) for runtime in client_runtime
                ),
                "max_cache_age": max(
                    (runtime.cache_age for runtime in client_runtime
                     if runtime.cache_age is not None), default=0
                ),
            }) if b3_plan else {})
            hist["telemetry"].append({**public_obs, **b3_telemetry, "action": action.type,
                                      "action_bundle": "|".join(action_types),
                                      "declared_bundle": "|".join(declared_types),
                                      "decision_bundle_status": decision_bundle_status,
                                      "raw_output": raw_output or "",
                                      "public_observation_json": observation_json,
                                      "full_public_observation_json": serialize_public_observation(public_obs),
                                      "agent_interface_error": interface_error,
                                      "activated_bundle": "|".join(activated_bundle.action_types),
                                      "emitted_at_round": r if decision_due else "n/a",
                                      "effective_from_round": r + 1 if decision_due else "n/a",
                                      "active_before": "|".join(transition.active_before) or "no_op",
                                      "active_after": "|".join(transition.active_after) or "no_op",
                                      "actions_started": "|".join(transition.started),
                                      "actions_stopped": "|".join(transition.stopped),
                                      "bundle_invalid": bundle_invalid,
                                      "decision_due": decision_due,
                                      "robust_variant": robust_info["variant"],
                                      "byzantine_frac": float(cfg.byzantine_frac),
                                      "compromised_num": int(robust_info["compromised_num"]),
                                      "robust_keep_n": int(robust_info["keep_n"]),
                                      "robust_selected_clients": " ".join(map(str, robust_info["selected"])),
                                      "trigger_source": robust_info["trigger_source"],
                                      "dropout_handle_substitutions": dropout_handle_count,
                                      "friend_substitutions": friend_count,
                                      "friend_similarity": round(float(np.mean(friend_sims)), 3) if friend_sims else 0.0,
                                      "friend_candidate_count": round(float(np.mean(friend_candidate_counts)), 3) if friend_candidate_counts else 0.0,
                                      "fdms_substitution_error": round(float(np.mean(friend_errors)), 6) if friend_errors else "n/a",
                                      "fdms_ready": bool(fdms_ready),
                                      "fdms_similarity_source": "update_cosine" if fdms_ready else "none",
                                      "cluster_purity": cluster_purity,
                                      "feddaa_enabled": bool(feddaa_enabled),
                                      "feddaa_mode": feddaa_mode,
                                      "feddaa_n_clusters": len(feddaa_models) if feddaa_models else 1,
                                      "feddaa_event": feddaa_event,
                                      "feddaa_split_score": round(float(feddaa_split_score), 6),
                                      "feddaa_label_history": bool(feddaa_enabled),
                                      "feddaa_cluster_source": feddaa_cluster_source,
                                      "feddaa_silhouette": round(float(feddaa_silhouette), 6),
                                      "feddaa_last_rebuild_round": (feddaa_last_rebuild_round
                                                                    if feddaa_last_rebuild_round is not None
                                                                    else "n/a"),
                                      "feddaa_report_clients": int(feddaa_report_clients),
                                      "feddaa_report_round": feddaa_report_round,
                                      "feddaa_report_bytes": int(feddaa_report_bytes),
                                      "feddaa_report_schema": 1,
                                      "feddaa_model_evaluations": int(feddaa_model_evaluations),
                                      "feddaa_prototype_shape": f"{self.data['n_classes']}x{self.data['n_classes']}",
                                      "feddaa_rebuild_count": int(feddaa_rebuild_count),
                                      "feddaa_assignment_changed_clients": int(feddaa_assignment_changed_clients),
                                      "feddaa_shift_clients": len(feddaa_shift_clients),
                                      "feddaa_clean_clients": len(feddaa_clean_clients),
                                      "feddaa_history_mix_clients": int(feddaa_history_mix_clients),
                                      "feddaa_sample_weight_entropy": round(float(feddaa_sample_weight_entropy), 6),
                                      "feddaa_cluster_sizes": " ".join(str(sum(1 for v in feddaa_assignment.values() if v == cid))
                                                                       for cid in range(len(feddaa_models) if feddaa_models else 1)),
                                      "feddaa_assignment_source": "prototype_silhouette_loss_label_weights" if feddaa_enabled else "none",
                                      "fedlc_enabled": bool(use_reweight and not feddaa_enabled),
                                      "fedlc_tau": float(getattr(cfg, "fedlc_tau", 1.0)),
                                      "fedlc_calibration_source": "online_client_histogram" if use_reweight and not feddaa_enabled else "none",
                                      "label_prior_adapt_enabled": bool(use_label_prior_adapt),
                                      "label_prior_source": label_prior_source,
                                      "label_prior_report_clients": int(label_prior_report_clients),
                                      "moment_align_enabled": bool(use_moment_align),
                                      "moment_align_clients": int(moment_align_clients),
                                      "moment_align_source": "global_input_mean_std" if use_moment_align else "none",
                                      "feddrift_num_models": len(spawn_models) if spawn_models else 1,
                                      "feddrift_num_concepts": len(set(spawn_assignment.values())) if spawn_assignment else 1,
                                      "feddrift_cluster_purity": cluster_purity,
                                      "feddrift_first_trigger_round": feddrift_first_trigger_round if feddrift_first_trigger_round is not None else "n/a",
                                      "feddrift_assignment_changed_clients": int(feddrift_assignment_changed_clients),
                                      "feddrift_max_concepts": int(cfg.feddrift_max_concepts),
                                      "feddrift_drifted_clients": " ".join(map(str, feddrift_drifted)),
                                      "feddrift_new_clusters": len(set(spawn_assignment.values())) if spawn_assignment else 1,
                                      "feddrift_cluster_threshold": float(getattr(cfg, "feddrift_h_deltap", cfg.feddrift_cluster_threshold)),
                                      "feddrift_h_delta": float(getattr(cfg, "feddrift_h_delta", cfg.feddrift_loss_delta)),
                                      "feddrift_h_deltap": float(getattr(cfg, "feddrift_h_deltap", cfg.feddrift_cluster_threshold)),
                                      "feddrift_marked_clients": len(feddrift_mark_until),
                                      "feddrift_merge_count": int(feddrift_merge_count),
                                      "feddrift_split_count": int(feddrift_split_count),
                                      "feddrift_reused_clients": int(feddrift_reused_clients),
                                      "feddrift_ari": feddrift_structure["ari"],
                                      "feddrift_nmi": feddrift_structure["nmi"],
                                      "feddrift_learner_count_error": feddrift_structure["learner_count_error"],
                                      "feddrift_sc_weight_total": round(float(sum(feddrift_sc_weights.values())), 3),
                                      "feddrift_sc_weight_bins": len(feddrift_sc_weights),
                                      "feddrift_assignment_records": len(feddrift_assignment_log),
                                      "feddrift_train_schedule": "compressed_sc_weights" if spawn_auto else "current_round",
                                      "feddrift_source_variant": "H_A_F_1_06_0" if spawn_auto else "none",
                                      "feddrift_assignment_source": "auto_hierarchical" if spawn_auto else ("oracle" if use_spawn else "single"),
                                      "using_drift_adapt": bool(use_drift_adapt),
                                      "using_retain_history_adapt": bool(use_retain_history_adapt),
                                      "using_reweight": bool(use_reweight),
                                      "using_label_prior_adapt": bool(use_label_prior_adapt),
                                      "weight_entropy": round(weight_entropy, 6),
                                      "max_weight": round(max_weight, 6),
                                      "reweight_source": reweight_source if use_reweight else "sample_count",
                                      "selected_by_action": bool(select_used),
                                      "selected_clients": " ".join(map(str, present)) if select_used else "",
                                      "select_invalid": int(select_invalid),
                                      "next_selected_clients": " ".join(map(str, select_action.clients)) if select_action else "",
                                      "reasoning": getattr(agent, "last_reasoning", ""),
                                      "diagnosis": getattr(agent, "last_diagnosis", ""),
                                      "llm_model": getattr(agent, "model_name", "n/a"),
                                      "llm_called": bool(getattr(agent, "called_this_round", False)),
                                      "llm_latency_ms": round(float(getattr(agent, "last_latency_ms", 0.0)), 3),
                                      "llm_prompt_tokens": int(getattr(agent, "last_prompt_tokens", 0)),
                                      "llm_completion_tokens": int(getattr(agent, "last_completion_tokens", 0)),
                                      "llm_cost_usd": round(float(getattr(agent, "last_cost_usd", 0.0)), 8),
                                      "llm_api_attempts": int(getattr(agent, "last_api_attempts", 0)),
                                      "llm_request_sha256": getattr(agent, "last_request_sha256", ""),
                                      "llm_replayed": bool(getattr(agent, "last_replayed", False)),
                                      "llm_parse_error": getattr(agent, "last_parse_error", ""),
                                      "llm_api_error": getattr(agent, "last_api_error", ""),
                                      "llm_budget_error": getattr(agent, "last_budget_error", "")})

            for action in round_actions:
                if action.type == "no_op":
                    continue
                if action.type not in {"spawn_concept", "spawn_concept_auto"} or staggered:
                    record_tool(action.type, r)
                if action.type == "lr_reset":
                    current_lr = cfg.lr0
                elif action.type == "select_clients":
                    selected_override = action.clients

            for action_type in transition.started:
                if ACTION_SPECS[action_type].lifecycle != "reversible_state":
                    continue
                if action_type in {"drift_adapt", "retain_history_adapt",
                                   "moment_align_adapt", "reweight"}:
                    current_lr = cfg.lr0

            for action_type in transition.one_shot:
                if action_type in ("spawn_concept", "spawn_concept_auto") and staggered:
                    if not use_spawn:
                        use_spawn = True
                        if action_type == "spawn_concept_auto":
                            spawn_auto = True
                            spawn_assignment = {ci: 1 for ci in range(n)}
                            feddrift_next_cluster_id = 2
                            cluster_purity = _feddrift_cluster_report(spawn_assignment, concept_of)
                        spawn_models = {}
                        model_ids = sorted(set(spawn_assignment.values())) if spawn_auto and spawn_assignment else range(1, cfg.n_concepts + 1)
                        for cid in model_ids:
                            spawn_models[cid] = clone_spawn_model(model)
                            feddrift_model_versions[cid] = 0
                            feddrift_model_weight[cid] = float(sum(len(y) for y in cy))
                        if spawn_auto and feddrift_pending_groups and len(spawn_models) < int(cfg.feddrift_max_concepts):
                            group = feddrift_pending_groups[0]
                            new_mid = feddrift_next_cluster_id
                            spawn_models[new_mid] = clone_spawn_model(model)
                            feddrift_model_versions[new_mid] = 0
                            feddrift_model_weight[new_mid] = 0.0
                            feddrift_next_cluster_id += 1
                            feddrift_split_count += 1
                            for ci in group:
                                spawn_assignment[ci] = new_mid
                                feddrift_mark_until[ci] = (
                                    new_mid, r + 1 + int(cfg.feddrift_mark_rounds)
                                )

            action_state = transition.state
            active_reversible = set(action_state.reversible)
            use_robust = "robust" in active_reversible
            use_dropout_handle = "dropout_handle" in active_reversible
            use_friend_substitute = "friend_substitute" in active_reversible
            use_fedprox = "fedprox" in active_reversible
            use_reweight = "reweight" in active_reversible
            use_label_prior_adapt = "label_prior_adapt" in active_reversible
            use_drift_adapt = "drift_adapt" in active_reversible
            use_retain_history_adapt = "retain_history_adapt" in active_reversible
            use_moment_align = "moment_align_adapt" in active_reversible
            feddaa_enabled = use_drift_adapt or use_retain_history_adapt
            feddaa_mode = "real" if use_drift_adapt else "virtual" if use_retain_history_adapt else ""

            current_lr = max(cfg.lr_floor, current_lr * cfg.lr_decay)
            if log:
                print(f"[{agent.name:14s}] r{r:02d} acc={acc:.3f} part={len(present)}/{n} "
                      f"-> {'+'.join(action_types)}")
        return hist
