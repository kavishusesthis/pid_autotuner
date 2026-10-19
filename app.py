"""app.py - interactive UI.  Run:  streamlit run app.py"""
import os, joblib, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
import core

st.set_page_config(page_title="PID Auto-Tuner", page_icon="⚙️", layout="wide")
st.title("⚙️ ML PID Auto-Tuner")
st.caption("Describe a DC motor → the ML model predicts Kp, Ki, Kd instantly → compare against classical tuning.")

if not os.path.exists("models/models.joblib"):
    st.warning("No trained model found. Run `python run_pipeline.py` first (takes a few minutes).")
    st.stop()
bundle = joblib.load("models/models.joblib"); model = bundle["models"][bundle["best"]]
tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Tuner", "📊 Benchmark", "🗂️ Dataset", "📖 How it works"])

with tab1:
    side, main = st.columns([1, 3])
    with side:
        st.subheader("Motor parameters")
        m = dict(R=st.slider("Resistance R (Ω)", .8, 4.0, 2.0, .1),
                 L=st.slider("Inductance L (mH)", 1.0, 8.0, 3.0, .1) / 1e3,
                 J=st.slider("Inertia J (×1e-4 kg·m²)", .5, 5.0, 2.0, .1) * 1e-4,
                 b=st.slider("Friction b (×1e-4 N·m·s)", .1, 5.0, 1.0, .1) * 1e-4,
                 Kt=st.slider("Torque const Kt (N·m/A)", .03, .12, .06, .005),
                 Ke=st.slider("Back-EMF const Ke (V·s/rad)", .03, .12, .06, .005))
        noise = st.slider("Sensor noise (% of target)", 0.0, 3.0, 0.0, .5) / 100
        run_opt = st.checkbox("Also run search optimizer (slower, ~0.1 s)", value=True)
        d = core.derived(m); st.metric("Mechanical time constant", f"{d['tau_m']*1e3:.0f} ms")
        if not .03 < d["tau_m"] < .5: st.warning("Outside the range the model was trained on - predictions may be poor.")
    G = {f"ML ({bundle['best']})": core.ml_predict(model, m), "Ziegler-Nichols": core.ziegler_nichols(m), "Cohen-Coon": core.cohen_coon(m)}
    if run_opt: G["Search optimizer"] = core.optimize(m)[0]
    r, t = core.setpoint(m), np.arange(1, int(core.T_END / core.TS) + 1) * core.TS
    with main:
        fig, ax = plt.subplots(figsize=(9, 4)); rows = []
        for n, g in G.items():
            y = core.simulate(m, *g, noise=noise); mt = core.metrics(y, r)
            ax.plot(t, y[:, 0], label=n, lw=2.5 if n.startswith("ML") else 1.3)
            rows.append({"Method": n, "Kp": g[0], "Ki": g[1], "Kd": g[2], "Rise (s)": mt["rise"][0], "Settle (s)": mt["settle"][0],
                         "Overshoot %": mt["overshoot"][0], "SS error %": mt["sse"][0], "Cost ↓": mt["cost"][0]})
        ax.axhline(r, c="k", ls=":", label="target"); ax.set_xlabel("time (s)"); ax.set_ylabel("speed (rad/s)"); ax.legend(); ax.grid(alpha=.3)
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(rows).set_index("Method").style.format("{:.4g}").highlight_min(subset=["Cost ↓"], color="#c8f7c5"), use_container_width=True)

with tab2:
    if os.path.exists("results/comparison_summary.csv"):
        st.subheader("Unseen-motor benchmark"); st.dataframe(pd.read_csv("results/comparison_summary.csv").set_index("method"), use_container_width=True)
        c1, c2 = st.columns(2); c1.image("results/comparison.png"); c2.image("results/step_responses.png")
        st.subheader("Robustness (median cost)"); st.dataframe(pd.read_csv("results/robustness.csv").set_index("method"), use_container_width=True)
        st.subheader("Model accuracy"); st.dataframe(pd.read_csv("results/model_scores.csv").set_index("model"), use_container_width=True); st.image("results/parity.png")
    else: st.info("Run `python run_pipeline.py` to generate benchmark results.")

with tab3:
    if os.path.exists("data/dataset.csv"):
        df = pd.read_csv("data/dataset.csv"); st.write(f"{len(df)} simulated motors, each labelled with its best-found PID."); st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "pid_dataset.csv")

with tab4:
    st.markdown("""
**Pipeline:**\n1) sample random DC motors (R, L, J, b, Kt, Ke) →\n2) for each, simulate ~600 PID combinations in closed loop (12 V limit, anti-windup)
→\n3) score each with a cost (rise + settling time, 2×overshoot, 5×steady-state error) →\n4) keep the best Kp/Ki/Kd →\n5) train Ridge / Random Forest / Gradient Boosting / MLP on
log-scaled targets →\n6) test on motors never seen in training and compare with Ziegler-Nichols, Cohen-Coon (open-loop FOPDT fit) and the search optimizer.

**Reading the results:** parameter R² is modest because many different PID triples give near-identical responses (a flat optimum);
the *control cost* is the fairer metric. The ML model gives near-optimizer quality at a fraction of the tuning time.\n
**Limitations:** simulated linear motor, speed control only, trained on a limited parameter range.""")
