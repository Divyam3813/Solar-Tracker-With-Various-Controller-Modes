"""Plant model, controller and the two tuning rules. python-control does the maths."""
import control as ct
import numpy as np

# ---- Motor and driver parameters (angles in degrees, so K is deg/s per volt) ----
RA, KT, KB = 2.5, 0.5, 0.5     # armature resistance, torque constant, back-EMF constant
J, B = 0.15, 0.005             # inertia (kg m^2) and tiny viscous friction
TAU_A = 0.5                    # driver lag (s), assumed
U_MAX = 5.0                    # voltage limit (V)
TAU_D = 0.05                   # derivative filter time constant (s)
CONTROLLERS = ("P", "PI", "PD", "PID")

_DEN = RA * B + KT * KB
K = KT / _DEN                  # plant gain     = Kt / (Ra*B + Kt*Kb)  ~ 1.905
TAU = RA * J / _DEN            # motor lag (s)  = Ra*J / (Ra*B + Kt*Kb) ~ 1.429

# ---- Transfer functions ----
s = ct.tf("s")
G = K / (s * (TAU * s + 1))    # motor + panel: voltage -> angle
GP = G / (TAU_A * s + 1)       # plus the driver lag


def pid(kp, ki=0.0, kd=0.0):
    """C(s) = Kp + Ki/s + Kd*s/(Td*s + 1). Unused terms are left out."""
    c = ct.tf([kp], [1])
    if ki:
        c = c + ki / s
    if kd:
        c = c + kd * s / (TAU_D * s + 1)
    return c


def closed_loop(kp, ki=0.0, kd=0.0):
    return ct.feedback(pid(kp, ki, kd) * GP)


def closed_loop_poles(kp, ki=0.0, kd=0.0):
    if not (kp or ki or kd):               # no controller: the library drops a zero transfer function
        return GP.poles()
    return closed_loop(kp, ki, kd).poles()


def is_stable(kp, ki=0.0, kd=0.0):
    return bool(closed_loop_poles(kp, ki, kd).real.max() < -1e-6)


# ---- Tuning rules: each returns {controller: (Kp, Ki, Kd)} ----
def ultimate_point():
    """Ziegler-Nichols data: gain Ku and period Tu at which P control just oscillates."""
    gain_margin, _, _, w_180, _, _ = ct.stability_margins(GP)
    return gain_margin, 2 * np.pi / w_180


def reaction_curve():
    """Cohen-Coon data: after a 1 V step the angle becomes a ramp of slope R that starts after delay L."""
    return K, TAU + TAU_A


def _gains(kp, Ti=None, Td=None):
    """Turn (Kp, Ti, Td) into the parallel gains (Kp, Ki = Kp/Ti, Kd = Kp*Td)."""
    return kp, (kp / Ti if Ti else 0.0), (kp * Td if Td else 0.0)


def zn_gains():
    """Ziegler-Nichols (closed-loop) table."""
    ku, tu = ultimate_point()
    return {"P": _gains(0.5 * ku),
            "PI": _gains(0.45 * ku, Ti=tu / 1.2),
            "PD": _gains(0.8 * ku, Td=tu / 8),
            "PID": _gains(0.6 * ku, tu / 2, tu / 8)}


def cohen_coon_gains():
    """Cohen-Coon (reaction-curve) table, in its limit for integrating plants."""
    r, L = reaction_curve()
    k = 1 / (r * L)
    return {"P": _gains(k),
            "PI": _gains(0.9 * k, Ti=10 * L / 3),
            "PD": _gains(1.25 * k, Td=3 * L / 11),
            "PID": _gains(4 / 3 * k, 32 * L / 13, 4 * L / 11)}