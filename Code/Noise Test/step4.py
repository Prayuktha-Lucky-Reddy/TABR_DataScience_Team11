%%writefile tabr_noise_experiment.py
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabr_all import (Block, download_data, find_datasets, get_args, load_raw, make_optimizer,
                      make_selftest_data, predict, preprocess, run_one, score_from_logits,
                      seed_everything)

NOISE_DATASETS = ["churn", "california", "house"]      # exact folder names


class MLP(nn.Module):
    """Same blocks as TabR (Linear -> residual Blocks -> head) but WITHOUT retrieval."""
    def __init__(self, d_in, d_out, d_main=265, d_multiplier=2.0, n_blocks=2,
                 dropout0=0.1, dropout1=0.0):
        super().__init__()
        d_block = int(d_main * d_multiplier)
        self.linear = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList(
            [Block(d_main, d_block, dropout0, dropout1, prenorm=(i > 0)) for i in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, d_out))

    def forward(self, x):
        x = self.linear(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


@torch.no_grad()
def predict_mlp(model, x_eval, bs):
    model.eval()
    return torch.cat([model(x_eval[i:i + bs]) for i in range(0, len(x_eval), bs)])


def train_mlp(data, seed, args, nargs, device):
    """Same recipe as run_one(): optimiser, lr, wd, batch-size rule, patience, val-based selection."""
    import copy
    seed_everything(seed)
    task = data["task"]
    d_out = data["n_classes"] if task == "multiclass" else 1
    d_in = data["x"]["train"].shape[1]
    model = MLP(d_in, d_out, args.d_main, args.d_multiplier, nargs.mlp_n_blocks,
                nargs.mlp_dropout, args.dropout1).to(device)
    x = {s: torch.tensor(v, device=device) for s, v in data["x"].items()}
    ydt = torch.float32 if task == "regression" else torch.long
    ym = {s: torch.tensor(v, dtype=ydt, device=device) for s, v in data["y_model"].items()}
    N = len(x["train"])
    bs = args.batch_size or (256 if N <= 20000 else 512 if N <= 100000 else 1024)
    eval_bs = max(bs * 2, 512)
    opt = make_optimizer(model, args)
    if task == "regression":
        loss_fn = lambda out, y: F.mse_loss(out.squeeze(-1), y)
    elif task == "binclass":
        loss_fn = lambda out, y: F.binary_cross_entropy_with_logits(out.squeeze(-1), y.float())
    else:
        loss_fn = lambda out, y: F.cross_entropy(out, y)

    best = {"score": -float("inf"), "state": None, "epoch": -1}
    bad, t0 = 0, time.time()
    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for step in range(-(-N // bs)):
            idx = perm[step * bs:(step + 1) * bs]
            loss = loss_fn(model(x["train"][idx]), ym["train"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_score, _ = score_from_logits(predict_mlp(model, x["val"], eval_bs), data["y"]["val"], data)
        if val_score > best["score"]:
            best.update(score=val_score, state=copy.deepcopy(model.state_dict()), epoch=epoch)
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
        if args.max_minutes and (time.time() - t0) > args.max_minutes * 60:
            break
    model.load_state_dict(best["state"])
    test_metric = score_from_logits(predict_mlp(model, x["test"], eval_bs), data["y"]["test"], data)[1]
    return model, eval_bs, {"test": test_metric, "epochs": best["epoch"] + 1, "time_s": time.time() - t0}


# ---------------- noise utilities ----------------
def make_unit_noise(n_rows, n_num, seed):
    """One fixed N(0,1) matrix; scaling it by each level gives identical, nested noise for both models."""
    return np.random.RandomState(seed).standard_normal((n_rows, n_num)).astype(np.float32)


def perturb_test_inputs(x_test, n_num, unit_noise, level, feat_std):
    """Noisy COPY of test tensor; only the first n_num (numerical) columns; categoricals untouched."""
    x = x_test.clone()
    if level > 0:
        x[:, :n_num] += level * feat_std[None, :] * unit_noise
    return x


def rmse_and_acc(logits, y_orig, data):
    task = data["task"]
    if task == "regression":
        return score_from_logits(logits, y_orig, data)[1], None
    if task == "binclass":  # RMSE of predicted probability vs 0/1 label (= sqrt Brier) + accuracy
        p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        return float(np.sqrt(np.mean((p - y_orig) ** 2))), float(((p > 0.5) == y_orig).mean())
    raise ValueError("noise experiment supports regression / binclass only")


def run_noise_experiment(name, path, args, nargs, device):
    raw = load_raw(path)
    data = preprocess(raw, args.num_policy)
    n_num, task = data["n_num"], data["task"]
    if n_num == 0:
        raise RuntimeError(f"{name}: no numerical features to perturb")
    print(f"\n=== {name} | task={task} | train={len(data['y']['train'])} "
          f"| test={len(data['y']['test'])} | num={n_num} | cat_onehot={data['n_cat_dim']}")
    x_test = torch.tensor(data["x"]["test"], device=device)
    y_test = data["y"]["test"]
    feat_std = torch.tensor(np.maximum(data["x"]["train"][:, :n_num].std(0), 1e-8), device=device)
    levels = [l / 100.0 for l in nargs.noise_levels]
    res = {"levels": nargs.noise_levels, "task": task,
           "tabr_rmse": [[] for _ in levels], "mlp_rmse": [[] for _ in levels],
           "tabr_acc": [[] for _ in levels], "mlp_acc": [[] for _ in levels], "clean_check": []}

    for seed in range(args.seeds):
        r = run_one(data, seed, args, device, name, return_model=True)    # TabR trained once
        tabr, tr, tabr_ebs = r["_model"], r["_tr"], r["_eval_bs"]
        mlp, mlp_ebs, mr = train_mlp(data, seed, args, nargs, device)     # MLP trained once
        unit_noise = torch.tensor(make_unit_noise(len(x_test), n_num, nargs.noise_seed + seed), device=device)
        for li, lv in enumerate(levels):
            xn = perturb_test_inputs(x_test, n_num, unit_noise, lv, feat_std)   # same xn for both models
            t_rmse, t_acc = rmse_and_acc(predict(tabr, xn, tr, tabr_ebs, args.context_size), y_test, data)
            m_rmse, m_acc = rmse_and_acc(predict_mlp(mlp, xn, mlp_ebs), y_test, data)
            res["tabr_rmse"][li].append(t_rmse); res["mlp_rmse"][li].append(m_rmse)
            res["tabr_acc"][li].append(t_acc);   res["mlp_acc"][li].append(m_acc)
            if lv == 0:  # must equal the normal clean evaluation from training
                ref_t, ref_m = r["test"], mr["test"]
                got_t = t_rmse if task == "regression" else t_acc
                got_m = m_rmse if task == "regression" else m_acc
                ok = abs(got_t - ref_t) < 1e-5 and abs(got_m - ref_m) < 1e-5
                res["clean_check"].append(bool(ok))
                print(f"  seed {seed}: 0% check vs clean eval -> TabR {got_t:.5f}/{ref_t:.5f}, "
                      f"MLP {got_m:.5f}/{ref_m:.5f}  {'OK' if ok else 'MISMATCH'}")
        print(f"  seed {seed} done | TabR best ep {r['epochs']} | MLP best ep {mr['epochs']}")
        del tabr, mlp, tr
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return res


# ---------------- tables / plots / interpretation ----------------
def _ms(v):
    return float(np.mean(v)), float(np.std(v))


def build_rows(res):
    rows = []
    m0, t0 = np.mean(res["mlp_rmse"][0]), np.mean(res["tabr_rmse"][0])
    for li, lv in enumerate(res["levels"]):
        mm, ms = _ms(res["mlp_rmse"][li]); tm, ts = _ms(res["tabr_rmse"][li])
        row = dict(noise_pct=lv, mlp_rmse=mm, mlp_std=ms, tabr_rmse=tm, tabr_std=ts,
                   mlp_rel_inc_pct=100 * (mm / m0 - 1), tabr_rel_inc_pct=100 * (tm / t0 - 1),
                   lower_rmse="TabR" if tm < mm else "MLP")
        if res["task"] == "binclass":
            row["mlp_acc"] = float(np.mean(res["mlp_acc"][li])); row["tabr_acc"] = float(np.mean(res["tabr_acc"][li]))
        rows.append(row)
    return rows


def format_table(rows, title):
    has_acc = "mlp_acc" in rows[0]
    h = f"{'noise%':>6} {'MLP RMSE':>16} {'TabR RMSE':>16} {'MLP +%':>9} {'TabR +%':>9} {'lower':>6}"
    if has_acc:
        h += f" {'MLP acc':>8} {'TabR acc':>9}"
    out = [title, h]
    for r in rows:
        s = (f"{r['noise_pct']:>6} {r['mlp_rmse']:>9.4f}±{r['mlp_std']:.4f} {r['tabr_rmse']:>9.4f}±{r['tabr_std']:.4f} "
             f"{r['mlp_rel_inc_pct']:>9.2f} {r['tabr_rel_inc_pct']:>9.2f} {r['lower_rmse']:>6}")
        if has_acc:
            s += f" {r['mlp_acc']:>8.4f} {r['tabr_acc']:>9.4f}"
        out.append(s)
    return "\n".join(out)


def save_plot(name, res, rows, out_dir):
    try:
        import matplotlib
        if "ipykernel" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped: {e})"); return
    x = [r["noise_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(x, [r["mlp_rmse"] for r in rows], yerr=[r["mlp_std"] for r in rows],
                marker="o", capsize=3, label="MLP")
    ax.errorbar(x, [r["tabr_rmse"] for r in rows], yerr=[r["tabr_std"] for r in rows],
                marker="s", capsize=3, label="TabR")
    ax.set_xlabel("Gaussian noise on numerical test features (% of feature std)")
    ax.set_ylabel("RMSE" + (" (of P(y=1))" if res["task"] == "binclass" else " (lower = better)"))
    ax.set_title(f"{name}: RMSE vs test-time noise ({len(res['mlp_rmse'][0])} seeds, ±1 std)")
    ax.set_xticks(x); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, f"noise_{name}.png")
    fig.savefig(p, dpi=150)
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)
    print(f"  saved {p}")


def interpret(all_res):
    """Auto-generated from the measured numbers (nothing hard-coded)."""
    lines, tabr_wins = [], 0
    for name, res in all_res.items():
        top = len(res["levels"]) - 1
        mr, tr_ = np.array(res["mlp_rmse"]), np.array(res["tabr_rmse"])
        m0, t0 = mr[0].mean(), tr_[0].mean()
        m_rel, t_rel = 100 * (mr[top].mean() / m0 - 1), 100 * (tr_[top].mean() / t0 - 1)
        m_abs, t_abs = mr[top].mean() - m0, tr_[top].mean() - t0
        d = 100 * (mr[top] / mr[0] - 1) - 100 * (tr_[top] / tr_[0] - 1)
        n_seeds = mr.shape[1]
        more_robust = "TabR" if t_rel < m_rel else "MLP"
        tabr_wins += more_robust == "TabR"
        if n_seeds < 3:
            conf = "only %d seed(s): indicative, not conclusive" % n_seeds
        elif abs(d.mean()) <= d.std():
            conf = "difference is within seed-to-seed variation: inconclusive"
        else:
            conf = "difference is larger than the seed-to-seed spread"
        lines.append(
            f"- {name}: clean RMSE MLP {m0:.4f} vs TabR {t0:.4f}. At {res['levels'][top]}% noise the RMSE rises "
            f"{m_rel:+.1f}% (abs {m_abs:+.4f}) for MLP and {t_rel:+.1f}% (abs {t_abs:+.4f}) for TabR "
            f"-> {more_robust} degrades more slowly in relative terms ({conf}).")
    n = len(all_res)
    lines.append(f"\nOverall: TabR degraded more slowly (relative RMSE increase at the highest noise level) on "
                 f"{tabr_wins} of {n} datasets, MLP on {n - tabr_wins}.")
    lines.append("Caveats: relative increase depends on the clean baseline, so also check absolute RMSE. MLP has "
                 "fixed untuned hyper-parameters, TabR uses its defaults; few seeds; noise only at test time. For "
                 "classification datasets RMSE is on predicted probabilities, not the paper's accuracy metric.")
    return "\n".join(lines)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = get_args(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--noise_levels", type=float, nargs="+", default=[0, 5, 10, 20, 30])
    p.add_argument("--noise_seed", type=int, default=12345)
    p.add_argument("--mlp_n_blocks", type=int, default=2)
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    p.add_argument("--noise_out_dir", default="noise_results")
    nargs, _ = p.parse_known_args(argv)
    assert nargs.noise_levels[0] == 0, "first noise level must be 0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.selftest:
        args.data_dir = "/tmp/tabr_selftest"
        make_selftest_data(args.data_dir)
        args.seeds, args.max_epochs, args.patience, args.d_main = 1, 8, 4, 64
    found = find_datasets(args.data_dir)
    if not found and not args.selftest:
        print("No datasets found, downloading ...")
        found = download_data("tabr_data")
    wanted = [n.lower() for n in (args.datasets or NOISE_DATASETS)]
    names = list(found) if args.selftest else [n for n in found if n.lower() in wanted]   # EXACT names
    missing = [w for w in wanted if w not in [n.lower() for n in found]]
    if missing and not args.selftest:
        print(f"WARNING: dataset folder(s) not found: {missing}")
    os.makedirs(nargs.noise_out_dir, exist_ok=True)

    all_res = {}
    for name in names:
        try:
            all_res[name] = run_noise_experiment(name, found[name], args, nargs, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !!! {name} failed: {type(e).__name__}: {e}")
    if not all_res:
        return

    combined = []
    for name, res in all_res.items():
        rows = build_rows(res)
        print("\n" + format_table(rows, f"--- {name} ({res['task']}) ---"))
        print(f"  0% matches clean evaluation for all seeds: {all(res['clean_check'])}")
        save_plot(name, res, rows, nargs.noise_out_dir)
        keys = list(rows[0].keys())
        with open(os.path.join(nargs.noise_out_dir, f"noise_{name}.csv"), "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        combined += [dict(dataset=name, task=res["task"], **r) for r in rows]

    print("\n=== COMBINED TABLE (mean RMSE over seeds) ===")
    print(f"{'dataset':<12}{'noise%':>7}{'MLP':>10}{'TabR':>10}{'MLP +%':>9}{'TabR +%':>9}{'lower':>7}")
    for r in combined:
        print(f"{r['dataset']:<12}{r['noise_pct']:>7}{r['mlp_rmse']:>10.4f}{r['tabr_rmse']:>10.4f}"
              f"{r['mlp_rel_inc_pct']:>9.2f}{r['tabr_rel_inc_pct']:>9.2f}{r['lower_rmse']:>7}")
    cols = ["dataset", "task", "noise_pct", "mlp_rmse", "mlp_std", "tabr_rmse", "tabr_std",
            "mlp_rel_inc_pct", "tabr_rel_inc_pct", "lower_rmse"]
    with open(os.path.join(nargs.noise_out_dir, "noise_combined.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in combined:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    text = interpret(all_res)
    print("\n=== INTERPRETATION ===\n" + text)
    with open(os.path.join(nargs.noise_out_dir, "noise_interpretation.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(nargs.noise_out_dir, "noise_raw.json"), "w") as f:
        json.dump(all_res, f)
    print(f"\nAll outputs saved in '{nargs.noise_out_dir}/'")
