"""
Sodium Channel Gating in the Hodgkin-Huxley Model
==================================================

Sodium channels open and close based on membrane voltage. The m gate
controls *activation* — it opens when the membrane depolarizes, allowing
sodium to rush in and drive the action potential upward.

The m gate is described by two voltage-dependent rate constants:
  alpha_m(V) : rate at which closed gates open
  beta_m(V)  : rate at which open gates close

From these we can derive:
  m_inf(V)  = alpha_m / (alpha_m + beta_m)   — steady-state open fraction
  tau_m(V)  = 1 / (alpha_m + beta_m)         — time constant (ms)
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SODIUM ACTIVATION GATE (m) RATE CONSTANTS
# =============================================================================

def alpha_m(V):
    """
    Opening rate for the m gate (1/ms).

    Uses L'Hopital's rule near V = -40 mV to avoid a 0/0 singularity.
    """
    dV = V + 40.0
    if np.isscalar(dV):
        if np.abs(dV) < 1e-7:
            return 1.0
        return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))
    result = np.zeros_like(dV, dtype=float)
    mask = np.abs(dV) < 1e-7
    result[mask] = 1.0
    result[~mask] = 0.1 * dV[~mask] / (1.0 - np.exp(-dV[~mask] / 10.0))
    return result


def beta_m(V):
    """Closing rate for the m gate (1/ms)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def m_inf(V):
    """Steady-state open fraction of the m gate."""
    am, bm = alpha_m(V), beta_m(V)
    return am / (am + bm)


def tau_m(V):
    """Time constant of the m gate (ms)."""
    return 1.0 / (alpha_m(V) + beta_m(V))


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_m_gate_kinetics():
    """
    Four-panel figure showing the full kinetics of the sodium m gate:
      - alpha_m(V) : opening rate
      - beta_m(V)  : closing rate
      - m_inf(V)   : steady-state open fraction
      - tau_m(V)   : time constant
    """
    V = np.linspace(-100, 60, 500)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('Sodium Channel m Gate Kinetics', fontsize=15, fontweight='bold')

    V_rest = -65.0  # reference resting potential

    # --- Panel 1: alpha_m ---
    ax = axes[0, 0]
    ax.plot(V, alpha_m(V), color='steelblue', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Rate (ms$^{-1}$)')
    ax.set_title(r'$\alpha_m(V)$ — Opening Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: beta_m ---
    ax = axes[0, 1]
    ax.plot(V, beta_m(V), color='tomato', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Rate (ms$^{-1}$)')
    ax.set_title(r'$\beta_m(V)$ — Closing Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: m_inf ---
    ax = axes[1, 0]
    ax.plot(V, m_inf(V), color='mediumseagreen', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.axhline(0.5, color='silver', linestyle=':', linewidth=1)
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Open fraction (0–1)')
    ax.set_title(r'$m_\infty(V)$ — Steady-State Open Fraction')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: tau_m ---
    ax = axes[1, 1]
    ax.plot(V, tau_m(V), color='darkorchid', linewidth=2)
    ax.axvline(V_rest, color='gray', linestyle='--', linewidth=1, label=f'V_rest = {V_rest} mV')
    ax.set_xlabel('Membrane Potential (mV)')
    ax.set_ylabel('Time constant (ms)')
    ax.set_title(r'$\tau_m(V)$ — Time Constant')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SODIUM CHANNEL m GATE KINETICS")
    print("=" * 60)

    V_rest = -65.0
    V_depol = 0.0

    print(f"""
The m gate controls sodium channel activation.

At rest (V = {V_rest} mV):
  alpha_m = {alpha_m(V_rest):.4f} ms⁻¹   (slow opening rate)
  beta_m  = {beta_m(V_rest):.4f} ms⁻¹   (fast closing rate)
  m_inf   = {m_inf(V_rest):.4f}          (gate nearly closed)
  tau_m   = {tau_m(V_rest):.4f} ms

During depolarization (V = {V_depol} mV):
  alpha_m = {alpha_m(V_depol):.4f} ms⁻¹   (fast opening rate)
  beta_m  = {beta_m(V_depol):.4f} ms⁻¹   (slow closing rate)
  m_inf   = {m_inf(V_depol):.4f}          (gate mostly open)
  tau_m   = {tau_m(V_depol):.4f} ms

The fast tau_m during depolarization means the m gate responds
quickly — sodium channels open within ~0.1 ms once voltage rises.
""")

    print("Generating plot...")
    plot_m_gate_kinetics()


if __name__ == "__main__":
    main()
