"""The sun, the cloud and the closed-loop simulation of the tracker."""
from functools import lru_cache
import control as ct
import numpy as np
from tuning import CONTROLLERS, GP, K, TAU_D, U_MAX

def cloud_factor(t, start, end, ramp=20.0):
    """0 = clear sky, 1 = full cloud cover. The cloud drifts in during `ramp` s after `start`
    and drifts out during the last `ramp` s before `end` (smooth S-curve)."""
    def smooth(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)
    ramp = max(min(ramp, (end - start) / 2), 1e-9)
    return smooth((t - start) / ramp) * smooth((end - t) / ramp)

@lru_cache(maxsize=None)
def _plant_steps(dt):
    """Exact discrete-time version of the plant: x[k+1] = A x[k] + B u[k], angle = C x[k]."""
    d = ct.sample_system(ct.tf2ss(GP), dt, "zoh")
    return d.A, d.B, d.C

def simulate(controller, kp, ki=0.0, kd=0.0, feedforward=False,
             cloud_start=20.0, cloud_end=120.0, ramp=20.0, cloud_severity=0.0,
             total_time=100.0, sweep=180.0, dt=0.02, seed=0):
    """Track a sun that moves `sweep` degrees in `total_time` seconds (a ramp). Returns time, sun, panel angle,
    error, motor voltage and a dict of metrics."""
    if controller not in CONTROLLERS:
        raise ValueError(f"controller must be one of {CONTROLLERS}")
    ki = ki if controller in ("PI", "PID") else 0.0        # gains the controller does not use are ignored
    kd = kd if controller in ("PD", "PID") else 0.0
    severity = min(max(cloud_severity, 0.0), 1.0)
    t = np.arange(round(total_time / dt)) * dt

    rate = sweep / total_time                              # sun speed (deg/s)
    sun = rate * t
    A, B, C = _plant_steps(dt)
    rng = np.random.default_rng(seed)

    x = np.zeros((3, 1))                                   # plant state
    theta, voltage = np.zeros(len(t)), np.zeros(len(t))
    integral = speed = 0.0
    for i in range(len(t)):
        theta[i] = (C @ x).item()
        error = sun[i] - theta[i]
        
        # Cloud: the light sensor sees a weaker and noisier error.
        cover = severity * cloud_factor(t[i], cloud_start, cloud_end)
        seen = (1 - 0.9 * cover) * error + (0.5 * cover * rng.standard_normal() if cover > 0 else 0.0)

        # Derivative of the error = sun speed - panel speed (panel speed from the encoder, filtered).
        if i:
            speed += dt / TAU_D * ((theta[i] - theta[i - 1]) / dt - speed)
        u_raw = kp * seen + ki * integral + kd * (rate - speed)
        if feedforward:
            u_raw += rate / K                              # voltage the known sun speed needs
        u = min(max(u_raw, -U_MAX), U_MAX)                 # voltage limit

        # Anti-windup: hold the integrator while the output is clipped and the error pushes it further.
        if u == u_raw or np.sign(seen) != np.sign(u_raw - u):
            integral += seen * dt
        voltage[i] = u
        x = A @ x + B * u

    error = sun - theta
    in_cloud = (t >= cloud_start) & (t <= cloud_end)
    metrics = {"IAE": round(float(np.abs(error).sum() * dt), 3),
               "ISE": round(float((error ** 2).sum() * dt), 3),
               "MaxError": round(float(np.abs(error).max()), 3),
               "CloudMaxError": round(float(np.abs(error[in_cloud]).max()), 3) if severity > 0 else 0.0,
               "FinalMeanError": round(float(np.abs(error[t >= t[-1] - 10]).mean()), 3),
               "SaturationPct": round(100 * float((np.abs(voltage) >= U_MAX - 1e-9).mean()), 1)}
    return {"t": t, "ref": sun, "theta": theta, "error": error, "u": voltage, "metrics": metrics}