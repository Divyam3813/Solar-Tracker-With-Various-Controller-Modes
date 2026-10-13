"""Plant model, PID controller and the two tuning rules (Ziegler-Nichols and Cohen-Coon).
The python-control library does the maths: transfer functions, feedback, poles, margins, step responses."""
import control as ct
import numpy as np

# ---- Motor and driver parameters (angle in degrees, so K is in deg/s per volt) ----
RA, KT, KB = 2.5, 0.5, 0.5     # armature resistance, torque constant, back-EMF constant
J, B = 0.15, 0.005             # inertia (kg m^2) and tiny viscous friction
TAU_A = 0.5                    # driver lag (s), assumed
U_MAX = 5.0                    # motor voltage limit (V)
TAU_D = 0.05                   # derivative filter time constant (s)

_DEN = RA * B + KT * KB
K = KT / _DEN                  # plant gain          = Kt / (Ra B + Kt Kb)
TAU = RA * J / _DEN            # motor time constant = Ra J / (Ra B + Kt Kb)

# ---- Plant: voltage -> panel angle ----
s = ct.tf("s")
GP = K / (s * (TAU * s + 1) * (TAU_A * s + 1))


def pid(kp, ki, kd):
    """C(s) = Kp + Ki/s + Kd s / (Tf s + 1). Terms that are zero are left out."""
    c = ct.tf([kp], [1])
    if ki:
        c = c + ki / s
    if kd:
        c = c + kd * s / (TAU_D * s + 1)
    return c


def closed_loop(kp, ki, kd):
    return ct.feedback(pid(kp, ki, kd) * GP)


def closed_loop_poles(kp, ki, kd):
    if not (kp or ki or kd):                 # no controller: the library cannot build a zero transfer function
        return GP.poles()
    return closed_loop(kp, ki, kd).poles()


def is_stable(kp, ki, kd):
    return bool(closed_loop_poles(kp, ki, kd).real.max() < -1e-6)


def to_gains(kp, ti, td):
    """Textbook form (Kp, Ti, Td) -> parallel gains (Kp, Ki, Kd) with Ki = Kp/Ti and Kd = Kp*Td."""
    return kp, kp / ti, kp * td


# ---- Ziegler-Nichols (closed-loop / ultimate-cycle method) ----
def ultimate_point():
    """Gain Ku and period Tu at which P control alone oscillates steadily.
    Found from the margins of the plant: Ku is the gain margin, Tu = 2*pi / (phase-crossover frequency)."""
    gain_margin, _, _, w_180, _, _ = ct.stability_margins(GP)
    return gain_margin, 2 * np.pi / w_180


def ziegler_nichols():
    """ZN PID rule:  Kp = 0.6 Ku,  Ti = Tu/2,  Td = Tu/8."""
    ku, tu = ultimate_point()
    return to_gains(0.6 * ku, tu / 2, tu / 8)


# ---- Cohen-Coon (open-loop / reaction-curve method) ----
def reaction_curve():
    """Step response of the plant, which ends as a ramp  y = R (t - L).
    R is the final slope and L is where that tangent meets the time axis (the dead time)."""
    t = np.linspace(0, 100, 10001)
    y = ct.step_response(GP, T=t).outputs
    r = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return r, t[-1] - y[-1] / r


def cohen_coon():
    """Cohen-Coon PID rule in the limit of an integrating process (time constant -> infinity):
        Kp = 4 / (3 R L),   Ti = 32 L / 13,   Td = 4 L / 11."""
    r, l = reaction_curve()
    return to_gains(4 / (3 * r * l), 32 * l / 13, 4 * l / 11)