"""run_pipeline.py - simulate motors -> label best PID -> train ML -> benchmark on unseen motors.
Usage:  python run_pipeline.py [--motors 1500] [--test 100] [--jobs -1]"""
import argparse, os, time, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import core

for d in ("data", "models", "results"): os.makedirs(d, exist_ok=True)


def label(m, i):
    g, c = core.optimize(m, seed=i); return {**m, "Kp": g[0], "Ki": g[1], "Kd": g[2], "best_cost": c}


def build(n, jobs):
    t = time.time()
    df = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(label)(m, i) for i, m in enumerate(core.sample_motors(n, 1))))
    df.to_csv("data/dataset.csv", index=False); print(f"[1/3] dataset: {n} motors in {time.time()-t:.0f}s"); return df


def train(df):
    X = np.array([core.features(r) for r in df.to_dict("records")]); Y = np.log10(df[["Kp", "Ki", "Kd"]].values)
    Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=.2, random_state=0)
    Xva, Xte, Yva, Yte = train_test_split(Xv, Yv, test_size=.5, random_state=0)
    zoo = {"Ridge": Ridge(1.0), "RandomForest": RandomForestRegressor(300, random_state=0),
           "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
           "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=0)}
    fit, rows = {}, []
    for n, mod in zoo.items():
        fit[n] = p = make_pipeline(StandardScaler(), mod).fit(Xtr, Ytr)
        rows.append(dict(model=n, val_MAE_log10=mean_absolute_error(Yva, p.predict(Xva)),
                         test_MAE_log10=mean_absolute_error(Yte, p.predict(Xte)), test_R2=r2_score(Yte, p.predict(Xte))))
    sc = pd.DataFrame(rows); best = sc.loc[sc.val_MAE_log10.idxmin(), "model"]
    sc.to_csv("results/model_scores.csv", index=False); joblib.dump(dict(models=fit, best=best), "models/models.joblib")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4)); P = fit[best].predict(Xte)
    for i, k in enumerate(["Kp", "Ki", "Kd"]):
        ax[i].scatter(Yte[:, i], P[:, i], s=12, alpha=.6); lim = [Yte[:, i].min(), Yte[:, i].max()]
        ax[i].plot(lim, lim, "r--"); ax[i].set_title(f"log10 {k}: predicted vs best"); ax[i].set_xlabel("best (search)"); ax[i].set_ylabel("ML")
    fig.suptitle(f"Held-out test motors - {best}"); fig.tight_layout(); fig.savefig("results/parity.png", dpi=110); plt.close(fig)
    print(f"[2/3] trained, best model = {best}\n{sc.round(3).to_string(index=False)}"); return joblib.load("models/models.joblib")


def evaluate(n, bundle):
    ms, ml = core.sample_motors(n, seed=999), bundle["models"][bundle["best"]]
    meth = {"ML": lambda m: core.ml_predict(ml, m), "Ziegler-Nichols": core.ziegler_nichols,
            "Cohen-Coon": core.cohen_coon, "Search optimizer": lambda m: core.optimize(m, seed=7)[0]}
    rows, gains = [], {}
    for i, m in enumerate(ms):
        r = core.setpoint(m)
        for name, f in meth.items():
            t = time.perf_counter(); g = f(m); dt = time.perf_counter() - t; gains[i, name] = g
            mt = core.metrics(core.simulate(m, *g), r)
            rows.append(dict(motor=i, method=name, time_s=dt, **{k: float(v[0]) for k, v in mt.items()}))
    d = pd.DataFrame(rows); d.to_csv("results/comparison_per_motor.csv", index=False)
    d["good"] = (d.overshoot < 10) & (d.settle < core.T_END - .05) & (d.sse < 2)
    s = d.groupby("method").agg(mean_cost=("cost", "mean"), median_cost=("cost", "median"), overshoot_pct=("overshoot", "mean"),
                                settle_s=("settle", "mean"), rise_s=("rise", "mean"), sse_pct=("sse", "mean"),
                                good_rate=("good", "mean"), time_ms=("time_s", lambda x: 1e3 * x.mean())).round(4)
    s.to_csv("results/comparison_summary.csv"); print("[3/3] benchmark on unseen motors\n", s.to_string())
    # robustness: sensor noise and +/-20% plant mismatch with gains tuned on the nominal motor
    rng, rob = np.random.default_rng(5), []
    for i, m in enumerate(ms[:40]):
        for name in meth:
            g = gains[i, name]
            for noise in (0, .01, .03):
                rob.append(dict(method=name, test=f"noise {noise:.0%}", cost=core.metrics(core.simulate(m, *g, noise=noise), core.setpoint(m))["cost"][0]))
            mm = {**m, "J": m["J"] * rng.uniform(.8, 1.2), "R": m["R"] * rng.uniform(.8, 1.2)}
            rob.append(dict(method=name, test="+/-20% J,R", cost=core.metrics(core.simulate(mm, *g), core.setpoint(mm))["cost"][0]))
    rb = pd.DataFrame(rob).pivot_table(index="method", columns="test", values="cost", aggfunc="median").round(4)
    rb.to_csv("results/robustness.csv"); print("robustness (median cost)\n", rb.to_string())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, i in zip(ax, (0, 1, 2)):
        m = ms[i]
        for name in meth: a.plot(np.arange(1, int(core.T_END / core.TS) + 1) * core.TS, core.simulate(m, *gains[i, name])[:, 0], label=name)
        a.axhline(core.setpoint(m), c="k", ls=":"); a.set_title(f"Unseen motor #{i}"); a.set_xlabel("time (s)"); a.set_ylabel("speed (rad/s)")
    ax[0].legend(); fig.tight_layout(); fig.savefig("results/step_responses.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); s.median_cost.plot.bar(ax=ax[0], title="Median cost (lower = better)")
    s.time_ms.plot.bar(ax=ax[1], logy=True, title="Tuning time per motor (ms)"); fig.tight_layout(); fig.savefig("results/comparison.png", dpi=110)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--motors", type=int, default=1500); a.add_argument("--test", type=int, default=100)
    a.add_argument("--jobs", type=int, default=-1); a = a.parse_args()
    evaluate(a.test, train(build(a.motors, a.jobs)))
