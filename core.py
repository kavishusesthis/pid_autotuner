"""core.py - DC-motor simulation, PID performance metrics, classical tuning rules."""
import numpy as np
from scipy.linalg import expm

TS, T_END, VMAX = 2e-3, 2.5, 12.0          # sample time (s), sim length (s), supply limit (V)
FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]  # R ohm, L H, J kg m^2, b N m s, Kt N m/A, Ke V s/rad


def derived(m):
    den = m["R"] * m["b"] + m["Ke"] * m["Kt"]
    return dict(K0=m["Kt"] / den, tau_m=m["J"] * m["R"] / den, tau_e=m["L"] / m["R"])


def setpoint(m):                              # target speed = 50% of the no-load speed (rad/s)
    return 0.5 * VMAX * derived(m)["K0"]


def plant(m):                                 # exact zero-order-hold discretisation of the motor
    A = np.array([[-m["R"] / m["L"], -m["Ke"] / m["L"]], [m["Kt"] / m["J"], -m["b"] / m["J"]]])
    M = np.zeros((3, 3)); M[:2, :2] = A * TS; M[0, 2] = TS / m["L"]
    E = expm(M); return E[:2, :2], E[:2, 2]


def simulate(m, kp, ki=None, kd=None, noise=0.0, seed=0):
    """Closed-loop step response. kp, ki, kd may be arrays -> many PID sets simulated at once."""
    kp, ki, kd = (np.atleast_1d(np.asarray(v, float)) for v in (kp, ki, kd))
    n, Ad, Bd, r = len(kp), *plant(m), setpoint(m)
    steps = int(T_END / TS); x = np.zeros((2, n)); I = np.zeros(n); prev = np.zeros(n)
    y = np.zeros((steps, n)); rng = np.random.default_rng(seed)
    for k in range(steps):
        w = x[1] + noise * r * rng.standard_normal(n)       # (noisy) speed sensor
        e = r - w
        raw = kp * e + I - kd * (w - prev) / TS              # derivative on measurement
        u = np.clip(raw, -VMAX, VMAX)
        wind = (raw != u) & (np.sign(raw) == np.sign(e))     # anti-windup
        I = np.clip(I + np.where(wind, 0, ki * e * TS), -VMAX, VMAX)
        prev = w; x = Ad @ x + Bd[:, None] * u; y[k] = x[1]
    return y


def metrics(y, r):
    t = np.arange(1, len(y) + 1) * TS; T = t[-1]
    os_ = np.clip(y.max(0) / r - 1, 0, None)
    sse = np.abs(y[-len(y) // 10:].mean(0) - r) / r
    up = y >= 0.9 * r; tr = np.where(up.any(0), t[up.argmax(0)], T)
    out = np.abs(y - r) > 0.02 * r
    last = len(y) - 1 - out[::-1].argmax(0)
    st = np.where(out.any(0), np.where(out[-1], T, t[np.minimum(last + 1, len(y) - 1)]), 0.0)
    cost = st / T + tr / T + 2 * os_ + 5 * sse               # lower = better
    return dict(rise=tr, settle=st, overshoot=100 * os_, sse=100 * sse,
                cost=np.nan_to_num(cost, nan=1e3, posinf=1e3))


# ---------- search-based "ground truth" tuner ----------
def gain_scale(m):
    d = derived(m); return np.array([1 / d["K0"], 1 / (d["K0"] * d["tau_m"]), d["tau_m"] / d["K0"]])


def sample_gains(m, n, rng, centre=None, width=0.25):
    lo, hi = np.log10([0.05, 0.1, 1e-4]), np.log10([300, 5000, 0.3])
    z = rng.uniform(lo, hi, (n, 3)) if centre is None else centre + rng.uniform(-width, width, (n, 3))
    return 10 ** z * gain_scale(m)


def optimize(m, n1=400, n2=200, seed=0):
    rng, r, s = np.random.default_rng(seed), setpoint(m), gain_scale(m)
    G = sample_gains(m, n1, rng); c = metrics(simulate(m, *G.T), r)["cost"]
    G2 = sample_gains(m, n2, rng, np.log10(G[c.argmin()] / s))          # zoom in around the best
    G, c = np.vstack([G, G2]), np.r_[c, metrics(simulate(m, *G2.T), r)["cost"]]
    return G[c.argmin()], c.min()


# ---------- classical rules (open-loop FOPDT fit, Smith 28.3% / 63.2% method) ----------
def fopdt(m):
    Ad, Bd = plant(m); x = np.zeros(2); ys = []
    for _ in range(int(T_END / TS)): x = Ad @ x + Bd; ys.append(x[1])
    ys, K = np.array(ys), derived(m)["K0"]; t = np.arange(1, len(ys) + 1) * TS
    t1, t2 = t[np.argmax(ys >= .283 * K)], t[np.argmax(ys >= .632 * K)]
    T = max(1.5 * (t2 - t1), TS); return K, max(t2 - T, TS), T


def ziegler_nichols(m):
    K, L, T = fopdt(m); kp = 1.2 * T / (K * L); return np.array([kp, kp / (2 * L), kp * .5 * L])


def cohen_coon(m):
    K, L, T = fopdt(m); a = L / T; kp = T / (K * L) * (4 / 3 + a / 4)
    return np.array([kp, kp / (L * (32 + 6 * a) / (13 + 8 * a)), kp * 4 * L / (11 + 2 * a)])


# ---------- ML helpers ----------
def sample_motors(n, seed=0):
    rng, out = np.random.default_rng(seed), []
    lu = lambda a, b: 10 ** rng.uniform(np.log10(a), np.log10(b))
    while len(out) < n:
        Kt = lu(.03, .12)
        m = dict(R=rng.uniform(.8, 4), L=lu(1e-3, 8e-3), J=lu(5e-5, 5e-4), b=lu(1e-5, 5e-4),
                 Kt=Kt, Ke=Kt * rng.uniform(.9, 1.1))
        if .03 < derived(m)["tau_m"] < .5: out.append(m)            # keep realistic dynamics
    return out


def features(m):
    d = derived(m); return np.log10([m[k] for k in FEATURES] + [d["K0"], d["tau_m"], d["tau_e"]])


def ml_predict(model, m):
    return 10 ** model.predict(features(m)[None])[0]
