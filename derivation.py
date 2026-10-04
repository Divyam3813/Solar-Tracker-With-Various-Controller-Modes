"""The 'Plant derivation' tab: how the transfer function is obtained, step by step."""
import pandas as pd
import streamlit as st

import tuning as tn


def render():
    den = tn.RA * tn.B + tn.KT * tn.KB
    st.header("How the plant transfer function is obtained")
    st.markdown("The signal path is **driver → motor + panel (voltage to speed) → integrator (speed to angle)**. "
                "Each block is derived below and the three are multiplied together.")
    st.markdown("##### Parameters used")
    st.table(pd.DataFrame([
        ["Ra", "Armature resistance", f"{tn.RA:g}", "Ω"],
        ["Kt", "Torque constant", f"{tn.KT:g}", "N·m/A"],
        ["Kb", "Back-EMF constant", f"{tn.KB:g}", "V/(rad/s)"],
        ["J", "Rotor + panel inertia", f"{tn.J:g}", "kg·m²"],
        ["B", "Viscous friction (tiny)", f"{tn.B:g}", "N·m/(rad/s)"],
        ["Ta", "Driver (actuator) lag, assumed", f"{tn.TAU_A:g}", "s"]],
        columns=["Symbol", "Meaning", "Value", "Unit"]).set_index("Symbol"))

    st.markdown("##### Step 1: electrical equation (armature inductance neglected)")
    st.latex(r"V(t) = R_a\, i(t) + K_b\,\omega(t)")
    st.caption("The applied voltage drives current through the resistance and must also overcome the back-EMF of the spinning motor.")
    st.markdown("##### Step 2: mechanical equation")
    st.latex(r"J\,\dot\omega(t) + B\,\omega(t) = K_t\, i(t)")
    st.caption("Motor torque Kt·i accelerates the inertia and fights viscous friction.")
    st.markdown("##### Step 3: Laplace transform (zero initial conditions)")
    st.latex(r"V(s) = R_a I(s) + K_b\,\Omega(s), \qquad (Js + B)\,\Omega(s) = K_t I(s)")
    st.markdown("##### Step 4: eliminate the current")
    st.latex(r"I(s)=\frac{(Js+B)\,\Omega(s)}{K_t}\;\Rightarrow\; V(s)=\frac{R_a(Js+B)+K_tK_b}{K_t}\,\Omega(s)"
             r"\;\Rightarrow\;\frac{\Omega(s)}{V(s)}=\frac{K_t}{R_aJ\,s+R_aB+K_tK_b}")
    st.markdown("##### Step 5: put it in standard first-order form (divide by Ra·B + Kt·Kb)")
    st.latex(r"\frac{\Omega(s)}{V(s)}=\frac{K}{\tau s+1},\qquad K=\frac{K_t}{R_aB+K_tK_b},\quad \tau=\frac{R_aJ}{R_aB+K_tK_b}")
    st.latex(rf"R_aB+K_tK_b = {tn.RA:g}\cdot{tn.B:g}+{tn.KT:g}\cdot{tn.KB:g} = {den:.4f}"
             rf"\;\Rightarrow\; K=\frac{{{tn.KT:g}}}{{{den:.4f}}}={tn.K:.4f},\quad \tau=\frac{{{tn.RA:g}\cdot{tn.J:g}}}{{{den:.4f}}}={tn.TAU:.4f}\ \mathrm{{s}}")
    st.markdown("##### Step 6: from speed to angle (integrator)")
    st.latex(r"\dot\theta=\omega\;\Rightarrow\;\Theta(s)=\frac{\Omega(s)}{s}\;\Rightarrow\;G(s)=\frac{\Theta(s)}{V(s)}=\frac{K}{s(\tau s+1)}")
    st.caption("The 1/s is why a constant voltage keeps the panel turning instead of holding an angle: the plant is Type 1.")
    st.markdown("##### Step 7: driver lag (first-order)")
    st.latex(r"T_a\,\dot u_a + u_a = u\;\Rightarrow\;\frac{U_a(s)}{U(s)}=\frac{1}{T_a s+1}")
    st.markdown("##### Step 8: multiply the blocks in series")
    st.latex(r"G_p(s)=\frac{1}{T_as+1}\cdot\frac{K}{\tau s+1}\cdot\frac{1}{s}=\frac{K}{s(\tau s+1)(T_as+1)}"
             r"=\frac{K}{\tau T_a s^3+(\tau+T_a)s^2+s}")
    st.latex(rf"G_p(s)=\frac{{{tn.K:.3f}}}{{{tn.TAU * tn.TAU_A:.4f}\,s^3+{tn.TAU + tn.TAU_A:.3f}\,s^2+s}}")
    st.markdown("##### Summary of results")
    st.table(pd.DataFrame([
        ["Ra·B + Kt·Kb", f"{den:.4f}", "common denominator"],
        ["K", f"{tn.K:.4f}", "deg/s per volt (angle in degrees, gearbox lumped in)"],
        ["τ", f"{tn.TAU:.4f} s", "motor + panel time constant"],
        ["Ta", f"{tn.TAU_A:g} s", "driver lag (assumed)"],
        ["Poles of Gp", f"0, {-1 / tn.TAU:.3f}, {-1 / tn.TAU_A:.3f}", "integrator, 1/τ, 1/Ta"]],
        columns=["Quantity", "Value", "Meaning"]).set_index("Quantity"))
    st.markdown("**Assumptions:** inductance neglected (electrical time ≪ mechanical time); linear viscous friction; rigid gearbox "
                "and panel; driver lag is assumed, not measured. On real hardware K and τ would be read from a step test of the motor.")