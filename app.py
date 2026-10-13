import base64
import time

import control as ct
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import chatbot
import derivation
import tuning as tn
from scene import MAX_CLOUDS, n_clouds, render_scene
from simulation import simulate

st.set_page_config(page_title="Solar Tracker PID Studio", page_icon="☀️", layout="wide")

# ---------------- Constants ----------------
METHODS = {"Ziegler-Nichols": tn.ziegler_nichols(), "Cohen-Coon": tn.cohen_coon()}    # each value is (Kp, Ki, Kd)
COLORS = {"Ziegler-Nichols": "tomato", "Cohen-Coon": "royalblue"}
KU, TU = tn.ultimate_point()
R_RC, L_RC = tn.reaction_curve()
DT = 0.02

st.title("☀️ Solar Tracker PID Studio")
st.markdown(f"Single-axis tracker following the sun from **0° (east) to 180° (west)** in 100 s · PID control · "
            f"Ziegler-Nichols and Cohen-Coon tuning · cloud disturbance on the light sensor · ±{tn.U_MAX:g} V limit")

# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("🎛️ PID controller")
method = sb.selectbox("Tuning method", [*METHODS, "Custom"])
kp0, ki0, kd0 = METHODS.get(method, (0.5, 0.1, 0.2))
kp = sb.slider("Kp", 0.0, 5.0, float(min(kp0, 5.0)), 0.01, key=f"kp_{method}")      # key changes with the method,
ki = sb.slider("Ki", 0.0, 2.0, float(min(ki0, 2.0)), 0.005, key=f"ki_{method}")     # so sliders reset to the preset
kd = sb.slider("Kd", 0.0, 3.0, float(min(kd0, 3.0)), 0.01, key=f"kd_{method}")
sb.markdown("---")
sb.subheader("☁️ Cloud")
cloud_sev = sb.slider("Cloud severity (0 = clear sky)", 0.0, 1.0, 0.6, 0.05)
cloud_start = sb.slider("Cloud arrives (s)", 5.0, 60.0, 30.0, 5.0)
cloud_end = cloud_start + sb.slider("Cloud duration (s)", 10.0, 60.0, 40.0, 5.0)
cloud = (cloud_sev, cloud_start, cloud_end)

# ---------------- Calculations (cached so moving a slider is fast) ----------------
@st.cache_data(show_spinner=False)
def run(kp, ki, kd, cloud):
    return simulate(kp, ki, kd, *cloud)


@st.cache_data(show_spinner=False)
def step_comparison():
    """Unit-step response of the closed loop for both tunings, plus rise time, overshoot, margins."""
    t = np.linspace(0, 60, 1200)
    curves, rows = {}, []
    for name, gains in METHODS.items():
        loop = tn.closed_loop(*gains)
        info = ct.step_info(loop, T=t)
        gm, pm = ct.stability_margins(tn.pid(*gains) * tn.GP)[:2]
        curves[name] = ct.step_response(loop, T=t).outputs
        rows.append({"Method": name, "Kp": round(gains[0], 3), "Ki": round(gains[1], 3), "Kd": round(gains[2], 3),
                     "Overshoot (%)": round(info["Overshoot"], 1), "Rise time (s)": round(info["RiseTime"], 2),
                     "Settling time (s)": round(info["SettlingTime"], 1), "Gain margin (x)": round(gm, 1),
                     "Phase margin (deg)": round(pm, 1), "Stable": tn.is_stable(*gains)})
    return t, curves, pd.DataFrame(rows).set_index("Method")


res = run(kp, ki, kd, cloud)
t, m = res["t"], res["metrics"]
worst_pole = float(tn.closed_loop_poles(kp, ki, kd).real.max())       # below 0 means the loop is stable


def style(fig, ytitle, height, title=None, shade=False):
    """Common look of the plots; optionally shade the cloud window."""
    if shade and cloud_sev > 0:
        fig.add_vrect(x0=cloud_start, x1=cloud_end, fillcolor="gray", opacity=0.25, layer="below",
                      line_width=0, annotation_text="☁️ Cloud", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title=ytitle, height=height,
                      margin=dict(l=20, r=20, t=30 if title is None else 50, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


tab_live, tab_sim, tab_der, tab_poles, tab_tuning, tab_chat = st.tabs(
    ["🌞 Live tracker", "📊 Simulation results", "🧮 Plant derivation", "📐 Poles & root locus",
     "🎚️ Tuning: ZN vs CC", "💬 Ask Dr. Adhyaru"])

# ---------------- Live tracker ----------------
with tab_live:
    c1, c2 = st.columns([5, 1])
    t_now = c1.slider("☀️ Move the sun (time, s)  ·  0 s = sunrise (0°), 100 s = sunset (180°)", 0.0, 100.0, 45.0, 0.5)
    play = c2.button("▶ Play")
    if worst_pole < -1e-6:
        st.success(f"Closed loop is stable (slowest pole real part = {worst_pole:.3f}).")
    else:
        st.error(f"Closed loop is UNSTABLE (pole real part = {worst_pole:.3f}).")

    def scene_html(seconds):
        i = min(int(round(seconds / DT)), len(t) - 1)
        svg = render_scene(float(t[i]), float(res["ref"][i]), float(res["theta"][i]),
                           cloud_sev, cloud_start, cloud_end, f"PID / {method}")
        return ('<img style="width:100%;border-radius:12px" src="data:image/svg+xml;base64,'
                + base64.b64encode(svg.encode()).decode() + '"/>')

    scene = st.empty()
    scene.markdown(scene_html(t_now), unsafe_allow_html=True)
    i = min(int(round(t_now / DT)), len(t) - 1)
    c = st.columns(4)
    c[0].metric("Sun angle (°)", f"{res['ref'][i]:.2f}")
    c[1].metric("Panel angle (°)", f"{res['theta'][i]:.2f}")
    c[2].metric("Pointing error (°)", f"{res['error'][i]:.2f}")
    c[3].metric("Motor command (V)", f"{res['u'][i]:.2f}")
    st.caption(f"Clouds drift in at t = {cloud_start:g} s and leave at t = {cloud_end:g} s. Severity {cloud_sev:.2f} shows "
               f"{n_clouds(cloud_sev)} of {MAX_CLOUDS} clouds and cuts the light-sensor signal by up to {90 * cloud_sev:.0f} %, "
               "plus noise. Move the sun slider into that window to see it.")
    if play:
        for seconds in range(100):
            scene.markdown(scene_html(seconds), unsafe_allow_html=True)
            time.sleep(0.03)
        scene.markdown(scene_html(t_now), unsafe_allow_html=True)

# ---------------- Simulation results (the gains chosen in the sidebar) ----------------
with tab_sim:
    st.markdown(f"##### PID with Kp = {kp:.3f}, Ki = {ki:.3f}, Kd = {kd:.3f}  ({method})")
    c = st.columns(6)
    c[0].metric("IAE", m["IAE"])
    c[1].metric("ISE", m["ISE"])
    c[2].metric("Max |error| (°)", m["MaxError"])
    c[3].metric("Max |error| in cloud (°)", m["CloudMaxError"] if cloud_sev > 0 else "–")
    c[4].metric("Mean |error|, last 10 s (°)", m["FinalMeanError"])
    c[5].metric("Saturated (%)", m["SaturationPct"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res["ref"], name="Sun", line=dict(color="orange", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=res["theta"], name="Panel angle", line=dict(color="dodgerblue", width=3)))
    fig.add_vline(x=t_now, line_dash="dot", line_color="white")
    st.markdown("### Tracking response")
    st.plotly_chart(style(fig, "Angle (°)", 380, shade=True))
    st.markdown("### Tracking error (sun − panel)")
    st.plotly_chart(style(go.Figure(go.Scatter(x=t, y=res["error"], line=dict(color="crimson", width=2))),
                          "Error (°)", 240, shade=True))
    st.markdown(f"### Motor command (limit ±{tn.U_MAX:g} V)")
    fig = go.Figure(go.Scatter(x=t, y=res["u"], line=dict(color="seagreen", width=2)))
    fig.add_hline(y=tn.U_MAX, line_dash="dot")
    fig.add_hline(y=-tn.U_MAX, line_dash="dot")
    st.plotly_chart(style(fig, "Voltage (V)", 240, shade=True))

# ---------------- Plant derivation (separate file) ----------------
with tab_der:
    derivation.render()

# ---------------- Poles and root locus ----------------
with tab_poles:
    c = st.columns(3)
    c[0].metric("K (deg/s per V)", f"{tn.K:.3f}")
    c[1].metric("τ (s)", f"{tn.TAU:.3f}")
    c[2].metric("Ta (s)", f"{tn.TAU_A:g}")
    st.markdown("##### Plant")
    st.latex(rf"G_p(s)=\frac{{{tn.K:.3f}}}{{s\,({tn.TAU:.3f}s+1)\,({tn.TAU_A:g}s+1)}}")
    st.table(pd.DataFrame([["s = 0", "0", "integrator: the angle accumulates the speed"],
                           ["s = -1/τ", f"{-1 / tn.TAU:.3f}", "motor + panel lag"],
                           ["s = -1/Ta", f"{-1 / tn.TAU_A:.3f}", "driver lag"]],
                          columns=["Pole", "Value", "Origin"]).set_index("Pole"))

    st.markdown("##### Root locus with P control: closed-loop poles as Kp increases from 0")
    gains = np.concatenate(([0.0], np.geomspace(1e-3, 3 * KU, 600)))
    locus = ct.root_locus_map(tn.GP, gains).loci                  # one row per gain, one column per branch
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=locus.real.ravel(), y=locus.imag.ravel(), mode="markers", name="closed-loop poles",
                             marker=dict(size=3, color=np.repeat(gains, locus.shape[1]), colorscale="Viridis",
                                         colorbar=dict(title="Kp"))))
    open_poles = tn.GP.poles()
    fig.add_trace(go.Scatter(x=open_poles.real, y=open_poles.imag, mode="markers", name="open-loop poles",
                             marker=dict(symbol="x", size=12, color="white")))
    at_ku = ct.root_locus_map(tn.GP, [KU]).loci[0]
    fig.add_trace(go.Scatter(x=at_ku.real, y=at_ku.imag, mode="markers", name=f"Kp = Ku = {KU:.3f}",
                             marker=dict(size=11, color="red")))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Real axis", yaxis_title="Imaginary axis", height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig)
    st.markdown(f"The locus crosses the imaginary axis at **Kp = Ku = {KU:.4f}**, where the poles are at "
                f"±j{2 * np.pi / TU:.3f} rad/s, so the oscillation period is **Tu = {TU:.4f} s**. "
                "These two numbers (read with `ct.stability_margins`) are the Ziegler-Nichols data.")

# ---------------- Tuning comparison ----------------
with tab_tuning:
    gains_table = pd.DataFrame(
        [[name, round(g[0], 4), round(g[1], 4), round(g[2], 4), round(g[0] / g[1], 3), round(g[2] / g[0], 3)]
         for name, g in METHODS.items()],
        columns=["Method", "Kp", "Ki = Kp/Ti", "Kd = Kp·Td", "Ti (s)", "Td (s)"]).set_index("Method")

    st.markdown("### 1. The two rules")
    a, b = st.columns(2)
    with a:
        st.markdown("**Ziegler-Nichols** (closed loop, ultimate cycle)")
        st.markdown("Raise the P gain until the output oscillates steadily. Read the gain **Ku** and period **Tu**.")
        st.latex(r"K_p=0.6\,K_u,\qquad T_i=\frac{T_u}{2},\qquad T_d=\frac{T_u}{8}")
        st.latex(rf"K_u={KU:.4f},\quad T_u={TU:.4f}\ \mathrm{{s}}")
    with b:
        st.markdown("**Cohen-Coon** (open loop, reaction curve)")
        st.markdown("Our plant integrates, so the step response ends as a ramp $y = R\\,(t - L)$. "
                    "Read the slope **R** and the dead time **L**. For a general first-order-plus-dead-time plant "
                    "(gain $K_p$, time constant $\\tau_p$, dead time $\\theta$) the PID rule is:")
        st.latex(r"K_c=\frac{\tau_p}{K_p\theta}\left(\frac43+\frac{\theta}{4\tau_p}\right),\quad "
                 r"T_i=\theta\,\frac{32+6\theta/\tau_p}{13+8\theta/\tau_p},\quad T_d=\frac{4\theta}{11+2\theta/\tau_p}")
        st.markdown("For an integrating plant $\\tau_p\\to\\infty$ while $R=K_p/\\tau_p$ stays fixed, so:")
        st.latex(r"K_p=\frac{4}{3RL},\qquad T_i=\frac{32L}{13},\qquad T_d=\frac{4L}{11}")
        st.latex(rf"R=K={R_RC:.4f}\ ^\circ/\mathrm{{s}}\,/\mathrm{{V}},\quad L=\tau+T_a={L_RC:.4f}\ \mathrm{{s}}")
    st.markdown("Both give the parallel PID  $C(s)=K_p+K_i/s+K_d\\,s/(T_f s+1)$  with  $K_i=K_p/T_i,\\ K_d=K_p T_d$:")
    st.table(gains_table)

    st.markdown("### 2. Where the tuning data comes from")
    a, b = st.columns(2)
    # Left: P control with Kp = Ku gives a steady oscillation of period Tu.
    t_long = np.linspace(0, 40, 2000)
    osc = ct.step_response(ct.feedback(KU * tn.GP), T=t_long).outputs
    fig = go.Figure(go.Scatter(x=t_long, y=osc, line=dict(color=COLORS["Ziegler-Nichols"], width=2), name="P control, Kp = Ku"))
    peaks = [i for i in range(1, len(osc) - 1) if osc[i] > osc[i - 1] and osc[i] >= osc[i + 1]]
    if len(peaks) >= 2:                          # mark one period between two late peaks
        p1, p2 = peaks[-2], peaks[-1]
        fig.add_annotation(x=t_long[p1], y=osc[p1], ax=t_long[p2], ay=osc[p2], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, text=f"Tu ≈ {t_long[p2] - t_long[p1]:.2f} s", yshift=14)
    a.plotly_chart(style(fig, "Angle (unit step)", 380, title="ZN: sustained oscillation at Kp = Ku"))
    # Right: open-loop step response is a ramp; its tangent meets the time axis at L.
    t_rc = np.linspace(0, 20, 1000)
    y_rc = ct.step_response(tn.GP, T=t_rc).outputs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_rc, y=y_rc, name="Plant step response (1 V)", line=dict(color=COLORS["Cohen-Coon"], width=3)))
    fig.add_trace(go.Scatter(x=t_rc, y=np.maximum(R_RC * (t_rc - L_RC), 0), name="Tangent: R (t − L)",
                             line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=[L_RC], y=[0], mode="markers+text", text=[f"L = {L_RC:.2f} s"], textposition="top center",
                             marker=dict(size=10, color="black"), showlegend=False))
    fig.update_yaxes(range=[-1, float(y_rc.max())])
    b.plotly_chart(style(fig, "Angle (°)", 380, title=f"CC: reaction curve, slope R = {R_RC:.3f} °/s"))

    st.markdown("### 3. Closed-loop step response of both tunings")
    t_step, curves, report = step_comparison()
    fig = go.Figure([go.Scatter(x=t_step, y=curves[name], name=f"{name} PID", line=dict(color=COLORS[name], width=3))
                     for name in METHODS])
    fig.add_hline(y=1, line_dash="dot")
    st.plotly_chart(style(fig, "Normalized angle", 400))
    st.dataframe(report)

    st.markdown("### 4. Tracking the sun ramp (using the cloud settings in the sidebar)")
    runs = {name: run(*g, cloud) for name, g in METHODS.items()}
    a, b = st.columns(2)
    fig = go.Figure([go.Scatter(x=t, y=runs[name]["error"], name=name, line=dict(color=COLORS[name], width=2)) for name in METHODS])
    a.plotly_chart(style(fig, "Error (°)", 360, title="Tracking error (sun − panel)", shade=True))
    fig = go.Figure([go.Scatter(x=t, y=runs[name]["u"], name=name, line=dict(color=COLORS[name], width=2)) for name in METHODS])
    fig.add_hline(y=tn.U_MAX, line_dash="dot")
    fig.add_hline(y=-tn.U_MAX, line_dash="dot")
    b.plotly_chart(style(fig, "Voltage (V)", 360, title="Motor command", shade=True))

    metric_names = {"IAE": "IAE", "ISE": "ISE", "MaxError": "Max |error| (°)", "CloudMaxError": "Max |error| in cloud (°)",
                    "FinalMeanError": "Mean |error|, last 10 s (°)"}
    table = pd.DataFrame({name: {label: runs[name]["metrics"][key] for key, label in metric_names.items()} for name in METHODS})
    a, b = st.columns([1, 1])
    a.dataframe(table)
    fig = go.Figure([go.Bar(x=["IAE", "ISE"], y=[runs[name]["metrics"]["IAE"], runs[name]["metrics"]["ISE"]],
                            name=name, marker_color=COLORS[name]) for name in METHODS])
    fig.update_layout(barmode="group", height=300, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Lower is better")
    b.plotly_chart(fig)
    st.info("ZN is faster and has the smaller tracking error, but it rings (large overshoot, small phase margin). "
            "Cohen-Coon is gentler, with lower gains and a bigger margin, but it lags the sun more.")

# ---------------- Chatbot ----------------
with tab_chat:
    st.subheader("💬 Ask Dr. Dipak Adhyaru (AI assistant)")
    st.info("This is an **AI assistant** built from Prof. Dipak Adhyaru's public profile (Nirma University). It is not the real "
            "professor, can make mistakes, and cannot speak for him. Check important answers with your guide.")
    with st.expander("About Dr. Dipak Adhyaru"):
        st.markdown(chatbot.PROFILE)
    api_key = chatbot.get_secret("GEMINI_API_KEY")
    if not api_key:
        st.warning("No Gemini key found. In Streamlit Cloud open **App → Settings → Secrets** and add:\n\n"
                   "`GEMINI_API_KEY = \"your-key\"`  (optional: `GEMINI_MODEL = \"model-name\"`)")
    state = {"Tuning": method, "Gains (Kp, Ki, Kd)": f"{kp:.3f}, {ki:.3f}, {kd:.3f}",
             "Cloud severity": f"{cloud_sev:.2f}", "Cloud window": f"{cloud_start:g}-{cloud_end:g} s",
             "Loop stable": worst_pole < -1e-6, "IAE": m["IAE"], "Max |error| (deg)": m["MaxError"],
             "Max |error| in cloud (deg)": m["CloudMaxError"], "Mean |error| last 10 s (deg)": m["FinalMeanError"],
             "K": round(tn.K, 4), "tau (s)": round(tn.TAU, 4), "Ta (s)": tn.TAU_A, "Ku": round(KU, 4), "Tu (s)": round(TU, 4)}
    history = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    examples = ["Why does the PID give zero steady error on the sun ramp?", "Explain Ziegler-Nichols vs Cohen-Coon here.",
                "Why does ZN overshoot more?", "Interpret my current results."]
    question = None
    for col, example in zip(st.columns(len(examples)), examples):
        if col.button(example, key=f"ex_{example}"):
            question = example
    question = st.chat_input("Ask about the plant, PID tuning, root locus...") or question
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not api_key:
                st.error("Add GEMINI_API_KEY to the app's secrets first.")
            else:
                model = chatbot.get_secret("GEMINI_MODEL")
                with st.spinner("Thinking..."):
                    answer, error = chatbot.ask(history, question, state, api_key, [model] if model else None)
                if error:
                    st.error(f"Gemini request failed: {error}")
                else:
                    st.markdown(answer)
                    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if history and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()