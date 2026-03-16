"""
Sodium Channel Inactivation Gating in the Hodgkin-Huxley Model
===============================================================

After sodium channels open (via the m gate), they don't stay open forever.
The h gate controls *inactivation* — it automatically closes after
depolarization, cutting off sodium current and preventing the action
potential from lasting too long.

The h gate behaves *opposite* to the m gate:
  - At rest (hyperpolarized), h is OPEN (near 1) — channels are ready to fire
  - During depolarization, h CLOSES (near 0) — channels become inactivated

The h gate is described by two voltage-dependent rate constants:
  alpha_h(V) : rate at which inactivated gates recover (reopen)
  beta_h(V)  : rate at which open gates become inactivated (close)

From these we can derive:
  h_inf(V)  = alpha_h / (alpha_h + beta_h)   — steady-state open fraction
  tau_h(V)  = 1 / (alpha_h + beta_h)         — time constant (ms)

Note: tau_h is much SLOWER than tau_m — inactivation lags behind activation,
which is what allows the action potential to occur in the first place.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SODIUM INACTIVATION GATE (h) RATE CONSTANTS
# =============================================================================

def alpha_h(V):
    """
    Recovery rate for the h gate (1/ms).

    At hyperpolarized voltages this is large — channels recover quickly.
    At depolarized voltages this is near zero — channels stay inactivated.
    """
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    """
    Inactivation rate for the h gate (1/ms).

    This is a sigmoid that rises steeply near -35 mV.
    At depolarized voltages, inactivation is fast.
    """
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def h_inf(V):
    """Steady-state open fraction of the h gate."""
    ah, bh = alpha_h(V), beta_h(V)
    return ah / (ah + bh)


def tau_h(V):
    """Time constant of the h gate (ms)."""
    return 1.0 / (alpha_h(V) + beta_h(V))


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_h_gate_kinetics():
    """
    Four-panel figure showing the full kinetics of the sodium h gate:
      - alpha_h(V) : recovery rate
      - beta_h(V)  : inactivation rate
      - h_inf(V)   : steady-state open fraction
      - tau_h(V)   : time constant
    """
    V = np.linspace(-100, 60, 500)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('Sodium Channel h Gate Kinetics (Inactivation)', fontsize=15, fontweight='bold')

    V_rest = -65.0  # reference resting potential

    # --- Panel 1: alpha_h ---
    ax = axes[0, 0]
    ax.plot(V, alpha_h(V), color='steelblue', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Rate (ms$^{-1}$)')
    ax.set_title(r'$\alpha_h(V)$ — Recovery Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: beta_h ---
    ax = axes[0, 1]
    ax.plot(V, beta_h(V), color='tomato', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Rate (ms$^{-1}$)')
    ax.set_title(r'$\beta_h(V)$ — Inactivation Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: h_inf ---
    ax = axes[1, 0]
    ax.plot(V, h_inf(V), color='darkorange', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.axhline(0.5, color='silver', linestyle=':', linewidth=1)
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Open fraction (0–1)')
    ax.set_title(r'$h_\infty(V)$ — Steady-State Open Fraction')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: tau_h ---
    ax = axes[1, 1]
    ax.plot(V, tau_h(V), color='darkorchid', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Time constant (ms)')
    ax.set_title(r'$\tau_h(V)$ — Time Constant')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SODIUM CHANNEL h GATE KINETICS (INACTIVATION)")
    print("=" * 60)

    V_rest = -65.0
    V_depol = 0.0

    print(f"""
The h gate controls sodium channel inactivation.
Unlike the m gate, h is OPEN at rest and CLOSES during depolarization.

At rest (V = {V_rest} mV):
  alpha_h = {alpha_h(V_rest):.4f} ms⁻¹   (slow recovery rate — not needed, gate already open)
  beta_h  = {beta_h(V_rest):.4f} ms⁻¹   (very slow inactivation rate)
  h_inf   = {h_inf(V_rest):.4f}          (gate mostly open — ready to fire)
  tau_h   = {tau_h(V_rest):.4f} ms

During depolarization (V = {V_depol} mV):
  alpha_h = {alpha_h(V_depol):.4f} ms⁻¹   (very slow recovery rate)
  beta_h  = {beta_h(V_depol):.4f} ms⁻¹   (fast inactivation rate)
  h_inf   = {h_inf(V_depol):.4f}          (gate nearly closed — channel inactivated)
  tau_h   = {tau_h(V_depol):.4f} ms

Notice:
  - tau_h at rest (~{tau_h(V_rest):.1f} ms) is much SLOWER than tau_m at rest
  - This lag is what makes the action potential possible:
    m opens fast, h closes slowly, so sodium can rush in before
    inactivation shuts the channel down.
  - The slow tau_h at depolarized voltages also sets the refractory period:
    after a spike, sodium channels stay inactivated for several ms.
""")

    print("Generating plot...")
    plot_h_gate_kinetics()


if __name__ == "__main__":
    main()
