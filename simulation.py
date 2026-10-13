"""The sun, the cloud and the closed-loop simulation of the tracker (PID control)."""
from functools import lru_cache

import control as ct
import numpy as np

from tuning import GP, TAU_D, U_MAX


def cloud_factor(t, start, end, ramp=20.0):
    """0 = clear sky, 1 = full cloud. The cloud drifts in during `ramp` s after `start`
    and drifts out during the last `ramp` s before `end` (smooth S-curve)."""
    def smooth(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)
    ramp = max(min(ramp, (end - start) / 2), 1e-9)
    return smooth((t - start) / ramp) * smooth((end - t) / ramp)


@lru_cache(maxsize=None)
def _plant_matrices(dt):
    """Exact discrete-time model of the plant:  x[k+1] = A x[k] + B u[k],  angle = C x[k]."""
    d = ct.sample_system(ct.tf2ss(GP), dt, "zoh")
    return d.A, d.B, d.C


def make_noise(n, seed=0):
    """Light-sensor noise: n standard-normal numbers. export_noise.py saves the same numbers for MATLAB/Simulink."""
    return np.random.default_rng(seed).standard_normal(n)


def simulate(kp, ki, kd, severity=0.0, cloud_start=30.0, cloud_end=70.0,
             total_time=100.0, sweep=180.0, dt=0.02, seed=0):
    """The sun moves `sweep` degrees in `total_time` seconds (a ramp) and the PID controller
    drives the panel after it. Returns time, sun angle, panel angle, error, motor voltage and metrics."""
    A, B, C = _plant_matrices(dt)
    t = np.arange(round(total_time / dt)) * dt
    sun = sweep / total_time * t                      # the ramp reference (deg)
    noise = make_noise(len(t), seed)                  # one noise sample per time step

    x = np.zeros((3, 1))                              # plant state
    theta = np.zeros(len(t))                          # panel angle
    volts = np.zeros(len(t))                          # motor voltage
    integral = 0.0                                    # integral of the error
    d_error = 0.0                                     # filtered derivative of the error
    last_error = 0.0

    for i in range(len(t)):
        theta[i] = (C @ x).item()
        error = sun[i] - theta[i]

        # A cloud makes the light sensor signal weaker and noisier (used by the P and I terms).
        cover = severity * cloud_factor(t[i], cloud_start, cloud_end)
        seen = (1 - 0.9 * cover) * error + 0.5 * cover * noise[i]

        # Derivative term: error rate through a first-order filter (taken from the encoder, not the sensor).
        if i > 0:
            d_error += dt / TAU_D * ((error - last_error) / dt - d_error)
        last_error = error

        u_raw = kp * seen + ki * integral + kd * d_error
        u = min(max(u_raw, -U_MAX), U_MAX)            # motor voltage limit
        if u == u_raw:                                # anti-windup: only integrate while not saturated
            integral += seen * dt

        volts[i] = u
        x = A @ x + B * u

    error = sun - theta
    in_cloud = (t >= cloud_start) & (t <= cloud_end)
    metrics = {"IAE": round(float(np.abs(error).sum() * dt), 2),
               "ISE": round(float((error ** 2).sum() * dt), 2),
               "MaxError": round(float(np.abs(error).max()), 2),
               "CloudMaxError": round(float(np.abs(error[in_cloud]).max()), 2) if severity > 0 else 0.0,
               "FinalMeanError": round(float(np.abs(error[t >= t[-1] - 10]).mean()), 3),
               "SaturationPct": round(100 * float((np.abs(volts) >= U_MAX - 1e-9).mean()), 1)}
    return {"t": t, "ref": sun, "theta": theta, "error": error, "u": volts, "metrics": metrics}