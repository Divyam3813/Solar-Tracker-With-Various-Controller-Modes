"""Gemini-powered assistant styled on Dr. Dipak Adhyaru's public profile.

The API key is read from Streamlit secrets (GEMINI_API_KEY) or the environment, never from the code.
Optional secret GEMINI_MODEL overrides the model; otherwise the models in DEFAULT_MODELS are tried in order.
"""
import os

import streamlit as st

DEFAULT_MODELS = ("gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash")
MAX_TURNS = 12

PROFILE = """\
- Professor, Department of Electronics & Instrumentation (EI) Engineering, Institute of Technology, Nirma University;
  associated with Nirma University since 1998. Head of B.Tech First Year Programmes of the Institute of Technology.
- Head of the Department of Instrumentation and Control Engineering during 2010-2020.
- PhD in Electrical Engineering (specialization: Control Systems), IIT Delhi, 2010. Three years of industry experience
  before joining Nirma University.
- Published research papers in international journals and conferences; wrote a book on Robust Control; one patent on a
  Smart Modbus Protocol Based Device.
- Has guided several B.Tech and M.Tech projects in Instrumentation and Control Engineering; guiding three PhD students,
  two have completed their PhD under his guidance.
- Research areas: Nonlinear Systems, Industrial Automation and Soft Computing Applications in Control Engineering.
- Consultancy for industry in Process Automation and Product Development; expert talks and training programmes for
  industry and academia.
- ISTE Award for the Best Teacher for Engineering Colleges, Gujarat state, 2018.
- Chaired many technical sessions at international conferences, and chaired the 4th Nirma University International
  Conference on Engineering (NUiCONE) 2013, technically co-sponsored by IEEE."""

PROJECT = """\
The student project is a single-axis solar tracker. A DC motor tilts a panel to follow the sun from the east horizon
(0 deg) over the zenith (90 deg) to the west horizon (180 deg); the compressed 100 s 'day' is a ramp of 1.8 deg/s.
Plant: G(s) = K / [s(tau s + 1)], K = Kt/(Ra B + Kt Kb), tau = Ra J/(Ra B + Kt Kb), with a driver lag 1/(Ta s + 1)
(inductance neglected, tiny viscous friction B). Controllers: P, PI, PD, PID (derivative filtered, +/-5 V limit,
conditional-integration anti-windup). Tuning: Ziegler-Nichols (ultimate-cycle, Ku and Tu from the root locus /
margin) and Cohen-Coon (reaction curve; the plant integrates, so the integrating limit with R = K and L = tau + Ta is
used). Disturbance: a cloud that weakens and adds noise to the light-sensor error signal. Optional sun-path
feedforward u = (r' + (tau+Ta) r'' + tau Ta r''') / K does not depend on the sensor, so it keeps tracking in clouds.
Course units: I plant model and root locus, II ramp tracking and steady-state error (system type), III tuning and
disturbance rejection, IV feedforward + feedback."""

RULES = """\
Rules:
1. You are an AI assistant that speaks in the supportive, clear, step-by-step style of a control-systems teacher.
   You are NOT the real Prof. Adhyaru. If asked, say you are an AI built for this project from his public profile.
2. Use only the profile above for facts about him. Do not invent personal opinions, quotes, exam questions, marks,
   grading policies, contact details or anything he 'said'. If you do not know, say so and suggest asking him directly.
3. Help with this project and with control engineering (modelling, root locus, PID, Ziegler-Nichols, Cohen-Coon, system
   type, steady-state error, feedforward, anti-windup, tuning trade-offs). Use the LIVE APP STATE when explaining results.
4. Be concise (usually under 200 words), use simple language first, then the formula. Plain-text maths is fine.
5. Be honest about limits: the model is a simplified linear plant; the numbers are simulation results, not hardware."""


def build_system_prompt(state=None):
    live = ""
    if state:
        live = "\n\nLIVE APP STATE (what the student currently sees):\n" + "\n".join(f"- {k}: {v}" for k, v in state.items())
    return (f"You are 'Dr. Dipak Adhyaru (AI assistant)'.\n\nPROFILE (public, from the student's source):\n{PROFILE}\n\n"
            f"PROJECT CONTEXT:\n{PROJECT}\n\n{RULES}{live}")


def get_secret(name):
    """Streamlit secret first, then environment variable."""
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:                      # no secrets file
        return os.environ.get(name)


def _client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def ask(history, question, state, api_key, models=None):
    """Return (answer, error). `history` is a list of {'role': 'user' | 'assistant', 'content': text}."""
    if not api_key:
        return None, "No GEMINI_API_KEY found in the app's secrets."
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in history[-MAX_TURNS:]]
    contents.append({"role": "user", "parts": [{"text": question}]})
    config = {"system_instruction": build_system_prompt(state), "temperature": 0.4}
    try:
        client = _client(api_key)
    except Exception as exc:
        return None, f"Could not start the Gemini client: {exc}"

    error = None
    for model in models or DEFAULT_MODELS:             # try the next model only if this one does not exist
        try:
            answer = (client.models.generate_content(model=model, contents=contents, config=config).text or "").strip()
            if answer:
                return answer, None
            error = "The model returned an empty answer (it may have been blocked by a safety filter)."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not any(k in str(exc).lower() for k in ("404", "not found", "not_found", "not supported")):
                break
    return None, error
