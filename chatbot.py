"""Gemini-powered assistant styled on Dr. Dipak Adhyaru's public profile.

The API key is read from Streamlit secrets (GEMINI_API_KEY) or the environment, never from the code.
Optional secret GEMINI_MODEL is tried first. After that the app asks the API which models your key can use,
ranks them fast-first, and falls back automatically when a model is busy (503), out of quota (429) or slow.
"""
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import streamlit as st

# Used only if the live model list cannot be fetched.
DEFAULT_MODELS = (
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
)
MAX_TURNS = 12

# ---- speed / fallback settings ----
HEDGE_S = 4.0            # if a model has not answered in 4 s, start the next one in parallel (first answer wins)
MAX_PARALLEL = 3         # never have more than 3 requests in flight
ATTEMPT_TIMEOUT_MS = 20_000   # hard stop for one request
TOTAL_BUDGET_S = 45.0    # give up after this long overall
MAX_CANDIDATES = 8       # how many models to try at most
COOLDOWN_BUSY_S = 60     # skip a model for 60 s after a 503 / timeout
COOLDOWN_QUOTA_S = 600   # skip a model for 10 min after a 429 (quota)
DISCOVERY_TTL_S = 3600
SKIP_WORDS = ("embedding", "tts", "image", "imagen", "live", "audio", "veo", "robotics", "computer-use",
              "aqa", "vision", "learnlm", "thinking-exp", "deep-research", "native")

_cooldown = {}                              # model -> time until which it is skipped
_discovered = {"t": 0.0, "models": []}
LAST_MODEL = None                           # which model answered last (handy for a caption in the UI)

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
Plant: Gp(s) = K / [s(tau s + 1)(Ta s + 1)], K = Kt/(Ra B + Kt Kb), tau = Ra J/(Ra B + Kt Kb), Ta = driver lag
(inductance neglected, tiny viscous friction B). Controller: PID only, C(s) = Kp + Ki/s + Kd s/(Tf s + 1)
(derivative filtered, +/-5 V limit, anti-windup by stopping the integrator while the output is clipped).
Tuning: Ziegler-Nichols closed-loop rule (Ku and Tu from the gain margin of the plant (Ku = (tau+Ta)/(K tau Ta), Tu = 2 pi sqrt(tau Ta));
Kp = 0.6 Ku, Ti = Tu/2, Td = Tu/8) and Cohen-Coon reaction-curve rule in its integrating-plant limit (R = K, L = tau + Ta;
Kp = 4/(3 R L), Ti = 32 L/13, Td = 4 L/11). Disturbance: a cloud that weakens and adds noise to the light-sensor error signal.
Course units: I plant model and root locus, II ramp tracking and steady-state error (system type), III tuning and
disturbance rejection."""

RULES = """\
Rules:
1. You are an AI assistant that speaks in the supportive, clear, step-by-step style of a control-systems teacher.
   You are NOT the real Prof. Adhyaru. If asked, say you are an AI built for this project from his public profile.
2. Use only the profile above for facts about him. Do not invent personal opinions, quotes, exam questions, marks,
   grading policies, contact details or anything he 'said'. If you do not know, say so and suggest asking him directly.
3. Help with this project and with control engineering (modelling, root locus, PID, Ziegler-Nichols, Cohen-Coon, system
   type, steady-state error, anti-windup, tuning trade-offs). Use the LIVE APP STATE when explaining results.
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
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=ATTEMPT_TIMEOUT_MS))


# ---------------------------------------------------------------- model discovery and ranking
def _rank(name):
    """Sort key: plain flash first, then flash-lite, then pro, then Gemma. Stable before preview, newer before older."""
    n = name.lower()
    if n.startswith("gemma"):
        tier = 4
    elif "pro" in n:
        tier = 3
    elif "lite" in n:
        tier = 2
    elif "flash" in n:
        tier = 0
    else:
        tier = 5
    preview = 1 if any(w in n for w in ("preview", "exp")) else 0
    not_alias = 0 if "latest" in n else 1                    # '-latest' aliases are stable, so try them first
    v = re.search(r"(\d+(?:\.\d+)?)", n)
    return (tier, preview, not_alias, -(float(v.group(1)) if v else 0.0), n)


def _available_models(client):
    """Models this key can call, best-first. Cached for an hour; falls back to DEFAULT_MODELS."""
    now = time.time()
    if _discovered["models"] and now - _discovered["t"] < DISCOVERY_TTL_S:
        return _discovered["models"]
    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            if not name.startswith(("gemini", "gemma")) or any(w in name for w in SKIP_WORDS):
                continue
            found.append(name)
    except Exception:
        found = []
    ordered = sorted(set(found), key=_rank) if found else list(DEFAULT_MODELS)
    _discovered.update(t=now, models=ordered)
    return ordered


def _candidates(client, preferred):
    seen, out = set(), []
    for m in [*(preferred or []), *_available_models(client)]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    now = time.time()
    ready = [m for m in out if _cooldown.get(m, 0) <= now]
    cooling = [m for m in out if m not in ready]               # recently failed models only as a last resort
    return (ready + cooling)[:MAX_CANDIDATES]


# ---------------------------------------------------------------- one request
def _error_kind(exc):
    text = f"{type(exc).__name__} {exc}"
    low = text.lower()
    if "429" in text or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "quota"
    if "api key" in low or "api_key_invalid" in low or "401" in text or "403" in text or "permission" in low:
        return "auth"
    if "404" in text or "not_found" in low or "not found" in low or "not supported" in low:
        return "missing"
    return "busy"                                              # 500/502/503/504, timeouts, empty answers, anything else


def _call(client, model, turns, system):
    from google.genai import types
    contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in turns]
    if model.startswith("gemma"):                              # Gemma models do not accept system instructions
        last = turns[-1][1]
        contents[-1] = types.Content(role="user", parts=[types.Part(text=f"{system}\n\nStudent question:\n{last}")])
        config = types.GenerateContentConfig(temperature=0.4)
    else:
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.4)
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty answer (possibly blocked by a safety filter)")
    return text


# ---------------------------------------------------------------- public function used by app.py
def ask(history, question, state, api_key, models=None):
    """Return (answer, error). `history` is a list of {'role': 'user' | 'assistant', 'content': text}.

    Starts with the best model. If it fails at once (503, 429, 404...) the next model starts immediately.
    If it is merely slow (HEDGE_S seconds) the next model starts in parallel and the first good answer wins.
    """
    global LAST_MODEL
    if not api_key:
        return None, "No GEMINI_API_KEY found in the app's secrets."
    try:
        client = _client(api_key)
    except Exception as exc:
        return None, f"Could not start the Gemini client: {exc}"

    system = build_system_prompt(state)
    turns = [("user" if m["role"] == "user" else "model", m["content"]) for m in history[-MAX_TURNS:]]
    turns.append(("user", question))

    queue = _candidates(client, list(models) if models else [])
    if not queue:
        return None, "No Gemini models available for this API key."

    errors, pending = [], {}
    pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
    deadline = time.time() + TOTAL_BUDGET_S

    def launch():
        model = queue.pop(0)
        pending[pool.submit(_call, client, model, turns, system)] = model

    try:
        launch()
        while pending and time.time() < deadline:
            done, _ = wait(list(pending), timeout=HEDGE_S, return_when=FIRST_COMPLETED)
            for fut in done:
                model = pending.pop(fut)
                try:
                    text = fut.result()
                except Exception as exc:                       # noqa: BLE001
                    kind = _error_kind(exc)
                    errors.append(f"{model}: {kind}")
                    if kind == "auth":
                        return None, f"API key problem: {exc}"
                    _cooldown[model] = time.time() + {"quota": COOLDOWN_QUOTA_S, "missing": 3600}.get(kind, COOLDOWN_BUSY_S)
                else:
                    LAST_MODEL = model
                    return text, None
            if queue and len(pending) < MAX_PARALLEL:          # a model failed or is slow: bring in the next one
                launch()
        for model in pending.values():                         # timed out while still waiting
            errors.append(f"{model}: timeout")
            _cooldown[model] = time.time() + COOLDOWN_BUSY_S
    finally:
        pool.shutdown(wait=False)                              # do not wait for slow leftover requests

    return None, ("All Gemini models are busy or out of quota right now. Tried: " + "; ".join(errors)
                  + ". Please try again in a minute.")
