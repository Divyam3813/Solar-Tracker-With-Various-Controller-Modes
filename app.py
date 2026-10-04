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

st.set_page_config(page_title="Solar Tracker Control Studio", page_icon="☀️", layout="wide")

# ---------------- Constants ----------------
METHODS = {"Ziegler-Nichols": tn.zn_gains(), "Cohen-Coon": tn.cohen_coon_gains()}   # {method: {controller: (Kp, Ki, Kd)}}
KU, TU = tn.ultimate_point()
R_RC, L_RC = tn.reaction_curve()
SUN_RATE = 180.0 / 100.0            # deg/s: east horizon to west horizon in 100 s
DT = 0.02

st.title("☀️ Solar Tracker Control Studio")
st.markdown(f"Single-axis tracker following the sun from **0° (east) to 180° (west)** · P / PI / PD / PID · "
            f"Ziegler-Nichols and Cohen-Coon tuning · cloud disturbance on the light sensor · ±{tn.U_MAX:g} V limit")

# ---------------- Sidebar: controller and cloud timing ----------------
sb = st.sidebar
sb.header("🎛️ Controller")
controller = sb.selectbox("Controller", tn.CONTROLLERS, index=3)
method = sb.selectbox("Tuning method", [*METHODS, "Custom"])
feedforward = sb.checkbox("Optional: add sun-path feedforward", value=False,
                          help="Commands the motor with the voltage the known sun path requires, "
                               "(θ̇ + (τ+Ta)θ̈ + τ·Ta·θ⃛)/K. It does not use the light sensor.")
default = METHODS[method][controller] if method in METHODS else (0.5, 0.1, 0.2)
uses_ki, uses_kd = controller in ("PI", "PID"), controller in ("PD", "PID")        # which gains this controller has
tag = f"{controller}_{method}"                                                    # new key = sliders reset to the preset
kp = sb.slider("Kp", 0.0, 5.0, float(min(default[0], 5.0)), 0.01, key=f"kp_{tag}")
ki = sb.slider("Ki", 0.0, 2.0, float(min(default[1], 2.0)), 0.005, key=f"ki_{tag}", disabled=not uses_ki)
kd = sb.slider("Kd", 0.0, 3.0, float(min(default[2], 3.0)), 0.01, key=f"kd_{tag}", disabled=not uses_kd)
ki, kd = (ki if uses_ki else 0.0), (kd if uses_kd else 0.0)
sb.markdown("---")
sb.subheader("☁️ Cloud event timing")
cloud_start = sb.slider("Cloud arrives (s)", 5.0, 60.0, 30.0, 5.0)
cloud_dur = sb.slider("Cloud duration (s)", 10.0, 60.0, 40.0, 5.0)
cloud_end = cloud_start + cloud_dur
sb.caption("Cloud severity and the sun slider are in the 'Live tracker' tab.")

(tab_live, tab_sim, tab_der, tab_poles, tab_tuning, tab_chat) = st.tabs(
    ["🌞 Live tracker", "📊 Simulation results", "🧮 Plant derivation", "📐 Poles & root locus",
     "🎚️ Tuning ZN vs CC", "💬 Ask Dr. Adhyaru"])

with tab_live:
    c1, c2, c3 = st.columns([3, 3, 1])
    t_now = c1.slider("☀️ Move the sun (time, s)  ·  0 s = sunrise (0°), 100 s = sunset (180°)", 0.0, 100.0, 45.0, 0.5, key="t_now")
    cloud_sev = c2.slider("☁️ Cloud severity (more clouds, weaker sensor)", 0.0, 1.0, 0.6, 0.05, key="cloud_sev")
    c3.write("")
    play = c3.button("▶ Play")


# ---------------- Simulations (cached, so moving a slider is fast) ----------------
@st.cache_data(show_spinner=False)
def run(ctrl, kp, ki, kd, ff, cloud):
    return simulate(ctrl, kp, ki, kd, ff, *cloud)                 # cloud = (start, end, severity)


@st.cache_data(show_spinner=False)
def compare_all(ff, cloud):
    """Simulate all 8 tunings."""
    rows = []
    for name, table in METHODS.items():
        for ctrl, g in table.items():
            m = run(ctrl, *g, ff, cloud)["metrics"]
            rows.append({"Method": name, "Ctrl": ctrl, "Kp": round(g[0], 3), "Ki": round(g[1], 3), "Kd": round(g[2], 3),
                         "Stable": tn.is_stable(*g), "IAE": m["IAE"], "ISE": m["ISE"], "Max err": m["MaxError"],
                         "Cloud max err": m["CloudMaxError"], "Last-10s |e|": m["FinalMeanError"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def step_report():
    """Linear step responses and margins of the 8 tunings (python-control)."""
    t = np.linspace(0, 60, 1200)
    curves, rows = {}, []
    for name, table in METHODS.items():
        for ctrl, g in table.items():
            loop = tn.closed_loop(*g)
            info = ct.step_info(loop, T=t)
            gm, pm, *_ = ct.stability_margins(tn.pid(*g) * tn.GP)
            curves[name, ctrl] = ct.step_response(loop, t).outputs
            rows.append({"Method": name, "Ctrl": ctrl, "Kp": round(g[0], 3), "Ki": round(g[1], 3), "Kd": round(g[2], 3),
                         "Stable": tn.is_stable(*g), "Overshoot %": round(info["Overshoot"], 1),
                         "Rise (s)": round(info["RiseTime"], 2), "Settling (s)": round(info["SettlingTime"], 1),
                         "GM": round(gm, 2), "PM (°)": round(pm, 1)})
    return t, curves, pd.DataFrame(rows)


cloud = (cloud_start, cloud_end, cloud_sev)
res = run(controller, kp, ki, kd, feedforward, cloud)
m, t = res["metrics"], res["t"]
worst = float(tn.closed_loop_poles(kp, ki, kd).real.max())          # <0 means the linear loop is stable
label = f"{controller}/{method[:2]}{'+FF' if feedforward else ''}"


def scene_html(seconds):
    i = min(int(round(seconds / DT)), len(t) - 1)
    svg = render_scene(float(t[i]), float(res["ref"][i]), float(res["theta"][i]), cloud_sev, cloud_start, cloud_end, label)
    return ('<img style="width:100%;border-radius:12px" src="data:image/svg+xml;base64,'
            + base64.b64encode(svg.encode()).decode() + '"/>')


def style(fig, ytitle, height, title=None, shade=False):
    """Common look of all plots; optionally shade the cloud window."""
    if shade and cloud_sev > 0:
        fig.add_vrect(x0=cloud_start, x1=cloud_end, fillcolor="gray", opacity=0.25, layer="below", line_width=0,
                      annotation_text="☁️ Cloud", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title=ytitle, height=height,
                      margin=dict(l=20, r=20, t=30 if title is None else 50, b=20))
    return fig


# ---------------- Live tracker ----------------
with tab_live:
    if worst < -1e-6:
        st.success(f"Linear closed loop is stable (slowest pole real part = {worst:.3f}).")
    else:
        st.error(f"Linear closed loop is UNSTABLE (pole real part = {worst:.3f}).")
    scene = st.empty()
    scene.markdown(scene_html(t_now), unsafe_allow_html=True)
    i = min(int(round(t_now / DT)), len(t) - 1)
    d = st.columns(4)
    d[0].metric("Sun angle (°)", f"{res['ref'][i]:.2f}")
    d[1].metric("Panel angle (°)", f"{res['theta'][i]:.2f}")
    d[2].metric("Pointing error (°)", f"{res['error'][i]:.2f}")
    d[3].metric("Motor command (V)", f"{res['u'][i]:.2f}")
    st.caption(f"Clouds drift in at t = {cloud_start:g} s and leave at t = {cloud_end:g} s. Severity {cloud_sev:.2f} shows "
               f"{n_clouds(cloud_sev)} of {MAX_CLOUDS} clouds and cuts the light-sensor signal by up to {90 * cloud_sev:.0f} %, "
               "plus noise. Move the sun slider into that window to see it.")
    if play:
        for seconds in range(100):
            scene.markdown(scene_html(seconds), unsafe_allow_html=True)
            time.sleep(0.03)
        scene.markdown(scene_html(t_now), unsafe_allow_html=True)

# ---------------- Simulation results ----------------
with tab_sim:
    c = st.columns(5)
    c[0].metric("IAE", m["IAE"]); c[1].metric("ISE", m["ISE"])
    c[2].metric("Max |error| (°)", m["MaxError"]); c[3].metric("Mean |error|, last 10 s (°)", m["FinalMeanError"])
    c[4].metric("Saturated (%)", m["SaturationPct"])
    c = st.columns(3)
    c[0].metric("Max |error| in cloud (°)", m["CloudMaxError"] if cloud_sev > 0 else "–")
    c[1].metric("Sun speed (°/s)", f"{SUN_RATE:.2f}")
    lag = SUN_RATE / (tn.K * kp) if (not uses_ki and kp > 0) else 0.0         # Type-1 loops lag a ramp by r/(K*Kp)
    c[2].metric("Predicted steady lag, feedback only", f"{lag:.2f}°" if not uses_ki else "0° (has integral)",
                help="Type-1 loops (P, PD) lag a ramp by r/(K·Kp). Feedforward removes most of it.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res["ref"], name="Sun", line=dict(color="orange", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=res["theta"], name="Panel angle", line=dict(color="dodgerblue", width=3)))
    fig.add_vline(x=t_now, line_dash="dot", line_color="white")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.markdown("### Tracking response")
    st.plotly_chart(style(fig, "Angle (°)", 400, shade=True))
    st.markdown("### True tracking error")
    st.plotly_chart(style(go.Figure(go.Scatter(x=t, y=res["error"], line=dict(color="crimson", width=2))), "Error (°)", 240, shade=True))
    st.markdown(f"### Motor command (limit ±{tn.U_MAX:g} V)")
    fig = go.Figure(go.Scatter(x=t, y=res["u"], line=dict(color="seagreen", width=2)))
    fig.add_hline(y=tn.U_MAX, line_dash="dot"); fig.add_hline(y=-tn.U_MAX, line_dash="dot")
    st.plotly_chart(style(fig, "Voltage (V)", 240, shade=True))

    st.markdown("### All 8 tunings with the current cloud and feedforward setting")
    st.dataframe(compare_all(feedforward, cloud), hide_index=True)

# ---------------- Plant derivation ----------------
with tab_der:
    derivation.render()

# ---------------- Poles and root locus ----------------
with tab_poles:
    c = st.columns(3)
    c[0].metric("K (deg/s per V)", f"{tn.K:.3f}"); c[1].metric("τ (s)", f"{tn.TAU:.3f}"); c[2].metric("Ta (s)", f"{tn.TAU_A:g}")
    st.caption("See the 'Plant derivation' tab for how these are obtained.")
    st.markdown("##### Transfer functions")
    st.table(pd.DataFrame(
        [["G(s)", "motor + panel", f"({tn.K:.4g}) / ({tn.TAU:.4g}s^2 + s)"],
         ["Gp(s)", "with driver lag", f"({tn.K:.4g}) / ({tn.TAU * tn.TAU_A:.4g}s^3 + {tn.TAU + tn.TAU_A:.4g}s^2 + s)"]],
        columns=["Name", "Meaning", "Expression"]).set_index("Name"))
    st.markdown("##### Poles of Gp(s)")
    st.table(pd.DataFrame([["s = 0", "0", "integrator: the angle accumulates the speed"],
                           ["s = -1/τ", f"{-1 / tn.TAU:.3f}", "motor + panel lag"],
                           ["s = -1/Ta", f"{-1 / tn.TAU_A:.3f}", "driver lag"]],
                          columns=["Pole", "Value", "Origin"]).set_index("Pole"))

    st.markdown("##### Root locus (closed-loop poles as the P gain Kp increases from 0)")
    cases = [(tn.G, np.concatenate(([0.0], np.geomspace(1e-3, 6.0, 500))), "G(s): stays in the left half-plane", False),
             (tn.GP, np.unique(np.concatenate(([0.0, KU], np.geomspace(1e-3, 3 * KU, 700)))),
              "Gp(s) with driver lag: crosses the imaginary axis at Ku", True)]
    for col, (system, gains, title, mark_ku) in zip(st.columns(2), cases):
        locus = ct.root_locus_map(system, gains).loci                 # one row per gain, one column per branch
        fig = go.Figure()
        for branch in locus.T:
            fig.add_trace(go.Scatter(x=branch.real, y=branch.imag, mode="lines", line=dict(width=2), showlegend=False))
        poles = system.poles()
        fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name="open-loop poles",
                                 marker=dict(symbol="x", size=11, color="white")))
        if mark_ku:
            at_ku = locus[int(np.argmin(np.abs(gains - KU)))]
            fig.add_trace(go.Scatter(x=at_ku.real, y=at_ku.imag, mode="markers", name=f"Kp = Ku = {KU:.3f}",
                                     marker=dict(size=10, color="red")))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(title=title, xaxis_title="Real axis", yaxis_title="Imaginary axis", height=400,
                          margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=-0.2))
        col.plotly_chart(fig)

    st.markdown("##### Ultimate point used by Ziegler-Nichols")
    closed_form = (tn.TAU + tn.TAU_A) / (tn.K * tn.TAU * tn.TAU_A), 2 * np.pi * np.sqrt(tn.TAU * tn.TAU_A)
    st.table(pd.DataFrame([["Imaginary-axis crossing of the root locus (python-control)", f"{KU:.4f}", f"{TU:.4f}"],
                           ["Closed-form (Routh): Ku = (τ+Ta)/(K·τ·Ta)", f"{closed_form[0]:.4f}", f"{closed_form[1]:.4f}"]],
                          columns=["Method", "Ku", "Tu (s)"]).set_index("Method"))

# ---------------- Tuning ----------------
with tab_tuning:
    st.markdown("##### Tuning rules")
    st.table(pd.DataFrame([["Ziegler-Nichols", "closed loop (ultimate cycle)", f"Ku = {KU:.3f}, Tu = {TU:.3f} s"],
                           ["Cohen-Coon", "open loop (reaction curve)", f"R = {R_RC:.3f} °/s per V, L = τ+Ta = {L_RC:.3f} s"]],
                          columns=["Method", "Type", "Data used"]).set_index("Method"))
    st.caption("Both rules were written for self-settling plants; this plant integrates, so Cohen-Coon uses its integrating limit. "
               "Step responses below are the linear model with derivative on the error, as in MATLAB pid().")
    t_step, curves, report = step_report()
    for col, name in zip(st.columns(2), METHODS):
        fig = go.Figure([go.Scatter(x=t_step, y=curves[name, ctrl], name=ctrl) for ctrl in tn.CONTROLLERS])
        fig.add_hline(y=1, line_dash="dot")
        col.plotly_chart(style(fig, "Normalized angle", 400, title=f"{name}: closed-loop step response"))
    st.dataframe(report, hide_index=True)
    st.info("ZN is faster and tracks the sun tighter but rings (ZN-PI especially); Cohen-Coon is better damped but slower.")

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
    state = {"Controller": controller, "Tuning": method, "Gains (Kp, Ki, Kd)": f"{kp:.3f}, {ki:.3f}, {kd:.3f}",
             "Sun-path feedforward": "on" if feedforward else "off", "Cloud severity": f"{cloud_sev:.2f}",
             "Cloud window": f"{cloud_start:g}-{cloud_end:g} s", "Linear loop stable": worst < -1e-6,
             "IAE": m["IAE"], "Max |error| (deg)": m["MaxError"], "Max |error| in cloud (deg)": m["CloudMaxError"],
             "Mean |error| last 10 s (deg)": m["FinalMeanError"], "K": round(tn.K, 4), "tau (s)": round(tn.TAU, 4),
             "Ta (s)": tn.TAU_A, "Ku": round(KU, 4), "Tu (s)": round(TU, 4)}
    history = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    examples = ["Why does P control leave a steady error on the sun ramp?", "Explain Ziegler-Nichols vs Cohen-Coon here.",
                "What does the sun-path feedforward do in a cloud?", "Interpret my current results."]
    question = None
    for col, example in zip(st.columns(len(examples)), examples):
        if col.button(example, key=f"ex_{example}"):
            question = example
    question = st.chat_input("Ask about the plant, tuning, root locus, feedforward...") or question
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