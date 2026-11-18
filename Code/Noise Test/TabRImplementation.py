%%writefile tabr_all.py
"""
TabR (Gorishniy et al., 2023) - ONE self-contained implementation that works on ANY
dataset of the paper's benchmark (regression / binary / multiclass).

The model class adapts itself to every dataset (feature counts, categorical one-hot size,
task type, number of classes). Weights are trained separately per dataset, exactly like
in the paper (TabR retrieves neighbours from each dataset's own training set).

Data format: the official one from https://github.com/yandex-research/tabular-dl-tabr
    data/<dataset_name>/info.json
    data/<dataset_name>/{X_num,X_cat,Y}_{train,val,test}.npy

Colab / Kaggle quick start (GPU runtime):
    !wget https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar -O data.tar
    !tar -xf data.tar
    !python tabr_all.py --data_dir data --seeds 3
Or let the script download the data:   !python tabr_all.py --download --seeds 3
List datasets only:                    !python tabr_all.py --data_dir data --list
Run chosen datasets (substring match): !python tabr_all.py --data_dir data --datasets california adult
Run every folder found (slow!):        !python tabr_all.py --all
Sanity check without any data:         !python tabr_all.py --selftest
(default = the 11 main datasets: churn, california, house, adult, diamond, otto, higgs-small, black-friday, weather-big, covtype, microsoft)

Metrics (same as the paper): RMSE for regression (original target scale, lower = better),
accuracy for binary / multiclass classification (higher = better).
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer, StandardScaler

DATA_URL = "https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar"


# ----------------------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_datasets(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "info.json"), recursive=True))
    return {os.path.basename(os.path.dirname(p)): os.path.dirname(p) for p in paths}


# Inside a notebook cell (Colab/Kaggle) command-line flags are not available, so edit this list instead,
# e.g. NOTEBOOK_ARGS = ["--datasets", "california", "adult", "--seeds", "3", "--max_minutes", "15"]
NOTEBOOK_ARGS = []

# the 11 datasets with plain folder names = the paper's main benchmark (ordered small -> large)
MAIN_DATASETS = ["churn", "california", "house", "adult", "diamond", "otto", "higgs-small",
                 "black-friday", "weather-big", "covtype", "microsoft"]


class SimpleAdamW:
    """Plain-PyTorch AdamW. Used only if torch.optim.AdamW cannot be created (see make_optimizer)."""

    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.wd, self.betas, self.eps, self.t = lr, weight_decay, betas, eps, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
            p.addcdiv_(m, denom, value=-self.lr / bc1)


_warned_opt = [False]


def make_optimizer(model, args):
    if not args.simple_adamw:
        try:
            return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        except Exception as e:  # e.g. broken torch._dynamo import in some Colab sessions
            if not _warned_opt[0]:
                print(f"NOTE: torch.optim.AdamW failed ({type(e).__name__}); using built-in AdamW instead "
                      f"(same algorithm). Restarting the runtime may also fix it.")
                _warned_opt[0] = True
    return SimpleAdamW(model.parameters(), args.lr, args.weight_decay)


def download_data(dest="tabr_data"):
    os.makedirs(dest, exist_ok=True)
    tar_path = os.path.join(dest, "data.tar")
    if not os.path.exists(tar_path):
        subprocess.check_call(["wget", "-q", DATA_URL, "-O", tar_path])
    subprocess.check_call(["tar", "-xf", tar_path, "-C", dest])
    found = find_datasets(dest)
    if not found:
        raise RuntimeError("archive extracted but no info.json files were found inside")
    return found


# ----------------------------------------------------------------------------------------
# data loading + preprocessing
# ----------------------------------------------------------------------------------------
def _load_npy(path, name, split):
    f = os.path.join(path, f"{name}_{split}.npy")
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def load_raw(path):
    with open(os.path.join(path, "info.json")) as f:
        info = json.load(f)
    raw = {"task": info["task_type"], "X_num": {}, "X_cat": {}, "Y": {}}
    for s in ["train", "val", "test"]:
        raw["X_num"][s] = _load_npy(path, "X_num", s)
        raw["X_cat"][s] = _load_npy(path, "X_cat", s)
        raw["Y"][s] = _load_npy(path, "Y", s)
    return raw


def preprocess(raw, num_policy="auto", seed=0):
    """Returns dict with float32 feature matrices [num | onehot(cat)], targets and metadata."""
    task = raw["task"]
    n_train = len(raw["Y"]["train"])
    out_x = {s: [] for s in ["train", "val", "test"]}
    n_num = 0

    # ---- numerical features
    if raw["X_num"]["train"] is not None:
        xs = {s: np.asarray(raw["X_num"][s], dtype=np.float64) for s in out_x}
        xs = {s: x.reshape(len(x), -1) for s, x in xs.items()}
        col_mean = np.nanmean(np.where(np.isfinite(xs["train"]), xs["train"], np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        for s in xs:  # fill NaN / inf with train mean
            bad = ~np.isfinite(xs[s])
            if bad.any():
                xs[s] = np.where(bad, col_mean[None], xs[s])
        policy = num_policy
        if policy == "auto":
            policy = "quantile"
        if policy == "quantile":
            rng = np.random.RandomState(seed)
            tr = xs["train"].copy()
            std = np.std(tr, axis=0, keepdims=True)
            noise = 1e-3  # tiny noise on train only (as in RTDL) to break ties
            tr = tr + (noise / np.maximum(std, noise)) * rng.randn(*tr.shape)
            qt = QuantileTransformer(
                n_quantiles=max(min(n_train // 30, 1000), 10),
                output_distribution="normal", subsample=10**9, random_state=seed)
            qt.fit(tr)
            xs = {s: qt.transform(x) for s, x in xs.items()}
        elif policy == "standard":
            sc = StandardScaler().fit(xs["train"])
            xs = {s: sc.transform(x) for s, x in xs.items()}
        else:
            raise ValueError(policy)
        for s in out_x:
            out_x[s].append(xs[s].astype(np.float32))
        n_num = xs["train"].shape[1]

    # ---- categorical features -> one-hot (categories fitted on train; unseen -> all zeros)
    n_cat_dim = 0
    if raw["X_cat"]["train"] is not None:
        cs = {s: np.asarray(raw["X_cat"][s]).astype(str) for s in out_x}
        cs = {s: c.reshape(len(c), -1) for s, c in cs.items()}
        blocks = {s: [] for s in out_x}
        for j in range(cs["train"].shape[1]):
            cats = np.unique(cs["train"][:, j])
            for s in out_x:
                blocks[s].append((cs[s][:, j][:, None] == cats[None, :]).astype(np.float32))
        for s in out_x:
            oh = np.concatenate(blocks[s], axis=1)
            out_x[s].append(oh)
        n_cat_dim = out_x["train"][-1].shape[1]

    x = {s: np.concatenate(v, axis=1) for s, v in out_x.items()}

    # ---- targets
    y = {s: np.asarray(raw["Y"][s]) for s in x}
    y_mean, y_std = 0.0, 1.0
    if task == "regression":
        y = {s: v.astype(np.float32).reshape(-1) for s, v in y.items()}
        y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
        y_std = y_std if y_std > 0 else 1.0
        y_model = {s: (v - y_mean) / y_std for s, v in y.items()}
        n_classes = None
    else:
        y = {s: v.astype(np.int64).reshape(-1) for s, v in y.items()}
        y_model = y
        n_classes = int(max(v.max() for v in y.values()) + 1)
        if task == "binclass":
            n_classes = 2
    return dict(x=x, y=y, y_model=y_model, task=task, n_num=n_num, n_cat_dim=n_cat_dim,
                n_classes=n_classes, y_mean=y_mean, y_std=y_std)


# ----------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------
class PLREmbeddings(nn.Module):
    """Periodic-Linear-ReLU embeddings for numerical features (optional)."""

    def __init__(self, n_features, n_frequencies=48, sigma=0.1, d_embedding=16):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.w = nn.Parameter(
            torch.randn(n_features, 2 * n_frequencies, d_embedding) / math.sqrt(2 * n_frequencies))
        self.b = nn.Parameter(torch.zeros(n_features, d_embedding))
        self.d_out = n_features * d_embedding

    def forward(self, x):  # (B, F)
        v = 2 * math.pi * self.c[None] * x[..., None]
        z = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)
        out = torch.relu(torch.einsum("bfk,fkd->bfd", z, self.w) + self.b)
        return out.flatten(1)


class Block(nn.Module):
    def __init__(self, d_main, d_block, dropout0, dropout1, prenorm):
        super().__init__()
        layers = []
        if prenorm:
            layers.append(nn.LayerNorm(d_main))
        layers += [nn.Linear(d_main, d_block), nn.ReLU(), nn.Dropout(dropout0),
                   nn.Linear(d_block, d_main), nn.Dropout(dropout1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class TabR(nn.Module):
    def __init__(self, n_num, n_cat_dim, d_out, task, n_classes=None,
                 d_main=265, d_multiplier=2.0, encoder_n_blocks=0, predictor_n_blocks=1,
                 context_dropout=0.38, dropout0=0.38, dropout1=0.0,
                 num_emb="none", plr_sigma=0.1, plr_freq=48, plr_dim=16):
        super().__init__()
        self.n_num = n_num
        self.task = task
        d_block = int(d_main * d_multiplier)

        # input layer (adapts to every dataset)
        self.num_emb = None
        d_in = n_cat_dim
        if n_num > 0:
            if num_emb == "plr":
                self.num_emb = PLREmbeddings(n_num, plr_freq, plr_sigma, plr_dim)
                d_in += self.num_emb.d_out
            else:
                d_in += n_num
        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [Block(d_main, d_block, dropout0, dropout1, prenorm=(i > 0))
             for i in range(encoder_n_blocks)])
        self.norm = nn.LayerNorm(d_main) if encoder_n_blocks > 0 else None

        # retrieval module
        self.K = nn.Linear(d_main, d_main)
        if task == "regression":
            self.label_enc = nn.Linear(1, d_main)
        else:
            self.label_enc = nn.Embedding(n_classes, d_main)
        self.T = nn.Sequential(nn.Linear(d_main, d_block), nn.ReLU(), nn.Dropout(dropout0),
                               nn.Linear(d_block, d_main, bias=False))
        self.ctx_drop = nn.Dropout(context_dropout)

        # predictor
        self.blocks1 = nn.ModuleList(
            [Block(d_main, d_block, dropout0, dropout1, prenorm=True)
             for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, d_out))

    def encode(self, x):
        if self.n_num > 0 and self.num_emb is not None:
            x = torch.cat([self.num_emb(x[:, :self.n_num]), x[:, self.n_num:]], dim=1)
        x = self.linear(x)
        for b in self.blocks0:
            x = b(x)
        k = self.K(x if self.norm is None else self.norm(x))
        return x, k

    @torch.no_grad()
    def all_keys(self, x_all, chunk=65536):
        ks = [self.encode(x_all[i:i + chunk])[1] for i in range(0, len(x_all), chunk)]
        return torch.cat(ks)

    @torch.no_grad()
    def search(self, k, cand_k, cand_sq, topk, exclude_mask=None, chunk=65536):
        B, N = k.shape[0], cand_k.shape[0]
        topk = min(topk, N)
        k2 = (k ** 2).sum(1, keepdim=True)
        best_d = torch.full((B, topk), float("inf"), device=k.device)
        best_i = torch.zeros((B, topk), dtype=torch.long, device=k.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = k2 - 2.0 * (k @ cand_k[s:e].T) + cand_sq[s:e][None]
            if exclude_mask is not None:
                d = d.masked_fill(exclude_mask[s:e][None], float("inf"))
            kk = min(topk, e - s)
            dv, di = d.topk(kk, dim=1, largest=False)
            all_d = torch.cat([best_d, dv], 1)
            all_i = torch.cat([best_i, di + s], 1)
            best_d, pos = all_d.topk(topk, dim=1, largest=False)
            best_i = all_i.gather(1, pos)
        return best_i

    def forward(self, x_in, cand_x, cand_y, cand_k, cand_sq, context_size, exclude_mask=None):
        x, k = self.encode(x_in)
        idx = self.search(k.detach(), cand_k, cand_sq, context_size, exclude_mask)  # (B, m)
        B, m = idx.shape
        ctx_x = cand_x[idx.reshape(-1)]
        ctx_k = self.encode(ctx_x)[1].reshape(B, m, -1)  # re-encoded WITH gradients
        ctx_y = cand_y[idx]
        if self.task == "regression":
            y_emb = self.label_enc(ctx_y[..., None])
        else:
            y_emb = self.label_enc(ctx_y)
        diff = k[:, None, :] - ctx_k
        values = y_emb + self.T(diff)
        sim = -(diff ** 2).sum(-1)
        probs = self.ctx_drop(torch.softmax(sim, dim=-1))
        x = x + (probs[:, None, :] @ values).squeeze(1)
        for b in self.blocks1:
            x = b(x)
        return self.head(x)


# ----------------------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------------------
def score_from_logits(logits, y_true_orig, data):
    task = data["task"]
    if task == "regression":
        pred = logits.squeeze(-1).cpu().numpy() * data["y_std"] + data["y_mean"]
        rmse = float(np.sqrt(np.mean((pred - y_true_orig) ** 2)))
        return -rmse, rmse  # (higher-is-better score, reported metric)
    if task == "binclass":
        pred = (logits.squeeze(-1) > 0).long().cpu().numpy()
    else:
        pred = logits.argmax(-1).cpu().numpy()
    acc = float((pred == y_true_orig).mean())
    return acc, acc


@torch.no_grad()
def predict(model, x_eval, tr, bs, context_size):
    model.eval()
    cand_k = model.all_keys(tr["x"])
    cand_sq = (cand_k ** 2).sum(1)
    outs = []
    for i in range(0, len(x_eval), bs):
        outs.append(model(x_eval[i:i + bs], tr["x"], tr["y"], cand_k, cand_sq, context_size))
    return torch.cat(outs)


def run_one(data, seed, args, device, name=""):
    seed_everything(seed)
    task = data["task"]
    d_out = data["n_classes"] if task == "multiclass" else 1
    model = TabR(
        n_num=data["n_num"], n_cat_dim=data["n_cat_dim"], d_out=d_out, task=task,
        n_classes=data["n_classes"], d_main=args.d_main, d_multiplier=args.d_multiplier,
        encoder_n_blocks=args.encoder_n_blocks, predictor_n_blocks=args.predictor_n_blocks,
        context_dropout=args.context_dropout, dropout0=args.dropout0, dropout1=args.dropout1,
        num_emb=args.num_emb, plr_sigma=args.plr_sigma, plr_freq=args.plr_freq,
        plr_dim=args.plr_dim).to(device)

    x = {s: torch.tensor(v, device=device) for s, v in data["x"].items()}
    ydt = torch.float32 if task == "regression" else torch.long
    ym = {s: torch.tensor(v, dtype=ydt, device=device) for s, v in data["y_model"].items()}
    tr = {"x": x["train"], "y": ym["train"]}
    N = len(x["train"])

    bs = args.batch_size or (256 if N <= 20000 else 512 if N <= 100000 else 1024)
    eval_bs = max(bs * 2, 512)
    ctx = args.context_size
    opt = make_optimizer(model, args)

    if task == "regression":
        loss_fn = lambda out, y: F.mse_loss(out.squeeze(-1), y)
    elif task == "binclass":
        loss_fn = lambda out, y: F.binary_cross_entropy_with_logits(out.squeeze(-1), y.float())
    else:
        loss_fn = lambda out, y: F.cross_entropy(out, y)

    best = {"score": -float("inf"), "state": None, "epoch": -1}
    bad_epochs, t0 = 0, time.time()
    steps_per_epoch = math.ceil(N / bs)
    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        cand_k = cand_sq = None
        for step in range(steps_per_epoch):
            idx = perm[step * bs:(step + 1) * bs]
            if cand_k is None or step % args.refresh_every == 0:
                cand_k = model.all_keys(tr["x"])
                cand_sq = (cand_k ** 2).sum(1)
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[idx] = True  # the whole batch is excluded from its own candidates
            out = model(x["train"][idx], tr["x"], tr["y"], cand_k, cand_sq, ctx, mask)
            loss = loss_fn(out, ym["train"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_logits = predict(model, x["val"], tr, eval_bs, ctx)
        val_score, val_metric = score_from_logits(val_logits, data["y"]["val"], data)
        if val_score > best["score"]:
            best.update(score=val_score, state=copy.deepcopy(model.state_dict()), epoch=epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.verbose and (epoch % 5 == 0):
            print(f"    [{name} s{seed}] ep {epoch:3d} loss {loss.item():.4f} "
                  f"val {val_metric:.4f} best_ep {best['epoch']} ({time.time() - t0:.0f}s)")
        if bad_epochs >= args.patience:
            break
        if args.max_minutes and (time.time() - t0) > args.max_minutes * 60:
            print(f"    time limit reached after {epoch + 1} epochs")
            break

    model.load_state_dict(best["state"])
    res = {}
    for s in ["val", "test"]:
        lg = predict(model, x[s], tr, eval_bs, ctx)
        res[s] = score_from_logits(lg, data["y"][s], data)[1]
    res["epochs"] = best["epoch"] + 1
    res["time_s"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------------------
# synthetic data for --selftest
# ----------------------------------------------------------------------------------------
def make_selftest_data(root):
    rng = np.random.RandomState(0)
    specs = [("synth_reg", "regression", 0), ("synth_bin", "binclass", 3), ("synth_multi", "multiclass", 2)]
    for name, task, n_cat in specs:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        sizes = {"train": 1500, "val": 400, "test": 400}
        for s, n in sizes.items():
            xn = rng.randn(n, 6).astype(np.float32)
            score = xn[:, 0] * 2 + np.sin(xn[:, 1] * 3) + 0.1 * rng.randn(n)
            np.save(os.path.join(d, f"X_num_{s}.npy"), xn)
            if n_cat:
                xc = rng.choice(["a", "b", "c", "d"], size=(n, n_cat)).astype(object)
                score = score + (xc[:, 0] == "a")
                np.save(os.path.join(d, f"X_cat_{s}.npy"), xc, allow_pickle=True)
            if task == "regression":
                y = score.astype(np.float32)
            elif task == "binclass":
                y = (score > np.median(score)).astype(np.int64)
            else:
                y = np.digitize(score, np.quantile(score, [0.33, 0.66])).astype(np.int64)
            np.save(os.path.join(d, f"Y_{s}.npy"), y)
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump({"task_type": task, "n_num_features": 6, "n_cat_features": n_cat}, f)


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--download", action="store_true", help="download + extract the official data")
    p.add_argument("--list", action="store_true", help="only list available datasets")
    p.add_argument("--datasets", nargs="*", default=None, help="name substrings to run")
    p.add_argument("--all", action="store_true", help="run ALL folders found (105 here); default = the 11 main datasets")
    p.add_argument("--simple_adamw", action="store_true", help="force the built-in AdamW")
    p.add_argument("--seeds", type=int, default=3, help="paper uses 15; 3-5 is enough to compare")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", default="tabr_results.csv")
    # model
    p.add_argument("--d_main", type=int, default=265)
    p.add_argument("--d_multiplier", type=float, default=2.0)
    p.add_argument("--encoder_n_blocks", type=int, default=0)
    p.add_argument("--predictor_n_blocks", type=int, default=1)
    p.add_argument("--context_dropout", type=float, default=0.38)
    p.add_argument("--dropout0", type=float, default=0.38)
    p.add_argument("--dropout1", type=float, default=0.0)
    p.add_argument("--context_size", type=int, default=96)
    p.add_argument("--num_emb", choices=["none", "plr"], default="none")
    p.add_argument("--plr_sigma", type=float, default=0.1)
    p.add_argument("--plr_freq", type=int, default=48)
    p.add_argument("--plr_dim", type=int, default=16)
    p.add_argument("--num_policy", choices=["auto", "quantile", "standard"], default="auto")
    # training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=0, help="0 = chosen from dataset size")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--refresh_every", type=int, default=1, help="recompute candidate keys every N steps")
    p.add_argument("--max_minutes", type=float, default=0, help="time limit per run (0 = none)")
    p.add_argument("--verbose", action="store_true")
    if argv is None and "ipykernel" in sys.modules:
        argv = NOTEBOOK_ARGS  # running inside a notebook cell
    a, _ = p.parse_known_args(argv)  # parse_known_args: ignores Jupyter/Colab's extra argv
    return a


def main(argv=None):
    args = get_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > GPU. "
              "In Kaggle: Settings > Accelerator > GPU.")

    if args.selftest:
        args.data_dir = "/tmp/tabr_selftest"
        make_selftest_data(args.data_dir)
        args.seeds, args.max_epochs, args.patience, args.d_main = 1, 8, 4, 64

    found = find_datasets(args.data_dir)
    if not found and not args.selftest:
        # look in the usual Colab / Kaggle places before giving up
        for root in [".", "/content", "/kaggle/input", "/kaggle/working"]:
            if os.path.isdir(root):
                found = find_datasets(root)
                if found:
                    print(f"found datasets under '{root}'")
                    break
    if not found and not args.selftest:
        # nothing found anywhere -> download the official data automatically
        print("No datasets found. Downloading the official data (this can take a few minutes)...")
        try:
            found = download_data("tabr_data")
        except Exception as e:
            print(f"Download failed: {e}\nOn Kaggle turn on Settings > Internet, or add the dataset "
                  f"manually and pass --data_dir <folder containing the dataset folders>.")
            return
    if not found:
        print(f"No datasets (info.json) found under '{args.data_dir}'. "
              f"Check the folder path with: !find . -name info.json | head")
        return
    names = list(found)
    if args.datasets:
        names = [n for n in names if any(q.lower() in n.lower() for q in args.datasets)]
    elif not args.all and not args.selftest:
        main_found = [n for n in MAIN_DATASETS if n in found]
        names = main_found if main_found else names
    if args.list:
        for n in found:
            print(n)
        return
    print(f"datasets to run ({len(names)}): {names}")

    rows = []
    for name in names:
        try:
            raw = load_raw(found[name])
            data = preprocess(raw, args.num_policy)
            print(f"\n=== {name} | task={data['task']} | train={len(data['y']['train'])} "
                  f"| num={data['n_num']} | cat_onehot={data['n_cat_dim']}"
                  f"{'' if data['n_classes'] is None else ' | classes=' + str(data['n_classes'])}")
            tests, vals = [], []
            for seed in range(args.seeds):
                r = run_one(data, seed, args, device, name)
                tests.append(r["test"])
                vals.append(r["val"])
                print(f"  seed {seed}: val {r['val']:.4f} | test {r['test']:.4f} "
                      f"| best epoch {r['epochs']} | {r['time_s']:.0f}s")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            metric = "RMSE (lower better)" if data["task"] == "regression" else "Accuracy (higher better)"
            rows.append((name, data["task"], metric, float(np.mean(tests)), float(np.std(tests)), len(tests)))
            print(f"  >>> {name}: {np.mean(tests):.4f} +- {np.std(tests):.4f}  [{metric}]")
        except Exception as e:  # keep going if one dataset fails
            if not getattr(main, "_tb_shown", False):
                import traceback
                traceback.print_exc()
                main._tb_shown = True
            print(f"  !!! {name} failed: {type(e).__name__}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'dataset':<28}{'task':<12}{'test mean':>12}{'std':>10}   metric")
    for n, t, m, mu, sd, k in rows:
        print(f"{n:<28}{t:<12}{mu:>12.4f}{sd:>10.4f}   {m}")
    with open(args.out, "w") as f:
        f.write("dataset,task,metric,test_mean,test_std,n_seeds\n")
        for n, t, m, mu, sd, k in rows:
            f.write(f"{n},{t},{m.split()[0]},{mu:.5f},{sd:.5f},{k}\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
