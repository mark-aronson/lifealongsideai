"""
Sodium Channel Activation: From Synaptic Pulses to Action Potential
====================================================================

This simulation delivers a series of brief synaptic current pulses of
increasing strength and tracks what happens to:

  1. The membrane voltage
  2. The fraction of open sodium channels (m³ · h)
  3. The intracellular sodium concentration

Key idea: there is a THRESHOLD. Weak pulses partially open sodium channels,
Na+ trickles in, and the membrane returns to rest. But once the pulse is
strong enough, the initial depolarization opens enough channels that their
combined sodium influx drives further depolarization — which opens even more
channels — a self-reinforcing cascade that produces the full action potential.

Below threshold: graded, passive response  →  membrane drifts back to rest
Above threshold: all-or-nothing avalanche  →  full ~+40 mV spike
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# PARAMETERS
# =============================================================================

# Membrane capacitance (µF/cm²)
C_m = 1.0

# Maximum conductances (mS/cm²)
g_Na_max = 120.0
g_K_max  = 36.0
g_L      = 0.3

# Reversal potentials (mV)
E_Na = 50.0
E_K  = -77.0
E_L  = -54.387

# Resting potential
V_rest = -65.0

# Physical constants (for Na concentration tracking)
F = 96485.0   # Faraday's constant (C/mol)
z = 1         # Na+ valence

# Simplified spherical cell geometry for Na concentration tracking
cell_diameter_um = 20.0                        # µm (typical soma)
cell_diameter_cm = cell_diameter_um * 1e-4     # cm
effective_depth_cm = cell_diameter_cm / 6.0   # volume/area ratio for a sphere = d/6
# This depth converts current density → concentration change

# Baseline intracellular Na concentration
Na_inside_baseline = 12.0   # mM

# Na/K pump: slowly restores [Na_in] to baseline after it rises
# Time constant ~400 ms (pumps restore Na between pulses)
tau_pump_ms = 400.0


# =============================================================================
# HODGKIN-HUXLEY GATING FUNCTIONS
# =============================================================================

def alpha_m(V):
    dV = V + 40.0
    if np.abs(dV) < 1e-7:
        return 1.0
    return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n(V):
    dV = V + 55.0
    if np.abs(dV) < 1e-7:
        return 0.1
    return 0.01 * dV / (1.0 - np.exp(-dV / 10.0))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def steady_state(V):
    """Steady-state gating variables at voltage V."""
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


# =============================================================================
# SIMULATION
# =============================================================================

def build_stimulus(t, pulse_times_ms, pulse_amplitudes, pulse_duration_ms):
    """
    Build a stimulus current array from a list of pulse onset times,
    amplitudes (µA/cm²), and a shared pulse duration.

    Parameters
    ----------
    t : array
        Time array (ms)
    pulse_times_ms : list of float
        Onset time of each pulse (ms)
    pulse_amplitudes : list of float
        Amplitude of each pulse (µA/cm²)
    pulse_duration_ms : float
        Duration of each pulse (ms)

    Returns
    -------
    I_stim : array
        Stimulus current at each timestep (µA/cm²)
    """
    I_stim = np.zeros_like(t)
    for t_on, amp in zip(pulse_times_ms, pulse_amplitudes):
        mask = (t >= t_on) & (t < t_on + pulse_duration_ms)
        I_stim[mask] = amp
    return I_stim


def run_threshold_simulation(
    pulse_times_ms,
    pulse_amplitudes,
    pulse_duration_ms=2.0,
    duration_ms=350.0,
    dt=0.01
):
    """
    Run a deterministic Hodgkin-Huxley simulation with a series of
    brief current pulses and track membrane voltage, sodium channel
    open fraction, and intracellular sodium concentration.

    Parameters
    ----------
    pulse_times_ms : list of float
        Onset times of stimulation pulses (ms)
    pulse_amplitudes : list of float
        Amplitudes of each pulse (µA/cm²)
    pulse_duration_ms : float
        Duration of each pulse (ms)
    duration_ms : float
        Total simulation time (ms)
    dt : float
        Integration timestep (ms)

    Returns
    -------
    dict with keys: t, V, open_frac, m, h, n, Na_in, I_stim
    """
    n_steps = int(duration_ms / dt)
    t = np.arange(n_steps) * dt

    # Build stimulus
    I_stim = build_stimulus(t, pulse_times_ms, pulse_amplitudes, pulse_duration_ms)

    # State arrays
    V       = np.zeros(n_steps)
    m_arr   = np.zeros(n_steps)
    h_arr   = np.zeros(n_steps)
    n_arr   = np.zeros(n_steps)
    Na_in   = np.zeros(n_steps)   # intracellular [Na] (mM)
    open_fr = np.zeros(n_steps)   # m³ · h

    # Initial conditions
    m0, h0, n0 = steady_state(V_rest)
    V[0], m_arr[0], h_arr[0], n_arr[0] = V_rest, m0, h0, n0
    Na_in[0] = Na_inside_baseline
    open_fr[0] = m0**3 * h0

    m, h, n = m0, h0, n0

    for i in range(n_steps - 1):
        Vi = V[i]

        # Conductances
        g_Na = g_Na_max * (m**3) * h
        g_K  = g_K_max  * (n**4)

        # Currents (µA/cm²)
        I_Na = g_Na * (Vi - E_Na)
        I_K  = g_K  * (Vi - E_K)
        I_L  = g_L  * (Vi - E_L)

        # Membrane voltage update (Euler)
        dV = (I_stim[i] - I_Na - I_K - I_L) / C_m
        V[i + 1] = Vi + dV * dt

        # Gate updates
        am, bm = alpha_m(Vi), beta_m(Vi)
        ah, bh = alpha_h(Vi), beta_h(Vi)
        an, bn = alpha_n(Vi), beta_n(Vi)

        m += (am * (1 - m) - bm * m) * dt
        h += (ah * (1 - h) - bh * h) * dt
        n += (an * (1 - n) - bn * n) * dt

        m_arr[i + 1] = m
        h_arr[i + 1] = h
        n_arr[i + 1] = n
        open_fr[i + 1] = m**3 * h

        # Intracellular Na concentration update
        # Inward I_Na (negative in HH convention) carries Na+ INTO the cell
        # Δ[Na_in] [mM] = -I_Na [µA/cm²] * dt [ms] * 1e-9 / (F [C/mol] * depth [cm]) * 1e6
        #               = -I_Na * dt * 1e-3 / (F * depth)
        delta_Na = -I_Na * dt * 1e-3 / (F * effective_depth_cm)
        # Na/K pump: exponential return to baseline
        pump = (Na_in[i] - Na_inside_baseline) * (dt / tau_pump_ms)
        Na_in[i + 1] = Na_in[i] + delta_Na - pump

    return {
        't':        t,
        'V':        V,
        'open_frac': open_fr,
        'm':        m_arr,
        'h':        h_arr,
        'n':        n_arr,
        'Na_in':    Na_in,
        'I_stim':   I_stim,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

# Color palette: cool-to-warm to show increasing pulse strength
PULSE_COLORS = ['#4575b4', '#74add1', '#f9a826', '#d73027', '#7b2d8b']


def plot_threshold_experiment(results, pulse_times_ms, pulse_amplitudes, pulse_duration_ms):
    """
    Four-panel figure illustrating the threshold phenomenon.

    Panel 1 — Stimulus: current pulse profile
    Panel 2 — Voltage: sub- vs supra-threshold membrane responses
    Panel 3 — Channel open fraction (m³·h): partial vs full sodium channel activation
    Panel 4 — Intracellular [Na]: small trickle vs large flood
    """
    t         = results['t']
    V         = results['V']
    open_frac = results['open_frac']
    Na_in     = results['Na_in']
    I_stim    = results['I_stim']

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        'Sodium Channel Activation:\nFrom Synaptic Pulses to Action Potential',
        fontsize=14, fontweight='bold'
    )

    n_pulses = len(pulse_times_ms)
    colors   = PULSE_COLORS[:n_pulses]

    # Detect which pulses triggered an action potential (V crosses +10 mV)
    ap_threshold_mV = 10.0
    fired = []
    for t_on, amp in zip(pulse_times_ms, pulse_amplitudes):
        # Look in a 50 ms window after each pulse onset
        window = (t >= t_on) & (t < t_on + 50.0)
        fired.append(np.any(V[window] > ap_threshold_mV))

    # ── Panel 1: Stimulus current ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(t, 0, I_stim, color='dimgray', alpha=0.7, step='post')
    ax1.set_ylabel('I$_{stim}$\n(µA/cm²)', fontsize=10)
    ax1.set_title('Stimulus: Brief Current Pulses of Increasing Strength', fontsize=11)

    # Label each pulse with its amplitude and colour
    for t_on, amp, color, did_fire in zip(pulse_times_ms, pulse_amplitudes, colors, fired):
        label = f'{amp:.0f} µA/cm²'
        if did_fire:
            label += '\n★ AP!'
        ax1.text(t_on + pulse_duration_ms / 2, amp * 1.12, label,
                 ha='center', va='bottom', fontsize=8.5,
                 color=color, fontweight='bold' if did_fire else 'normal')
        # Colour the pulse bar
        mask = (t >= t_on) & (t < t_on + pulse_duration_ms)
        ax1.fill_between(t[mask], 0, I_stim[mask], color=color, step='post')

    ax1.set_ylim([0, max(pulse_amplitudes) * 1.45])
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Membrane voltage ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, V, 'k-', linewidth=1.2)
    ax2.axhline(V_rest, color='steelblue', linestyle=':', linewidth=1,
                alpha=0.7, label=f'Rest ({V_rest:.0f} mV)')
    ax2.axhline(ap_threshold_mV, color='tomato', linestyle='--', linewidth=1,
                alpha=0.8, label=f'AP criterion ({ap_threshold_mV:.0f} mV)')

    # Shade sub- and supra-threshold regions
    ax2.fill_between(t, V_rest, V,
                     where=V > V_rest, interpolate=True,
                     color='salmon', alpha=0.25)

    ax2.set_ylabel('Membrane\nPotential (mV)', fontsize=10)
    ax2.set_title('Membrane Voltage — Graded Responses vs All-or-Nothing Action Potential', fontsize=11)
    ax2.set_ylim([-80, 55])
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Sodium channel open fraction ─────────────────────────────
    ax3 = axes[2]
    ax3.plot(t, open_frac, color='crimson', linewidth=1.2)

    # Mark resting open fraction
    m0, h0, _ = steady_state(V_rest)
    open_rest = m0**3 * h0
    ax3.axhline(open_rest, color='steelblue', linestyle=':', linewidth=1,
                alpha=0.7, label=f'Resting open fraction ({open_rest:.5f})')

    ax3.set_ylabel('Na channel\nopen fraction\n(m³ · h)', fontsize=10)
    ax3.set_title('Sodium Channel Gating — Partial vs Full Activation', fontsize=11)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.25)

    # Shade the "avalanche" region during any AP
    ax3.fill_between(t, 0, open_frac, color='crimson', alpha=0.15)

    # ── Panel 4: Intracellular sodium concentration ───────────────────────
    ax4 = axes[3]
    ax4.plot(t, Na_in, color='darkorange', linewidth=1.4)
    ax4.axhline(Na_inside_baseline, color='steelblue', linestyle=':', linewidth=1,
                alpha=0.7, label=f'Baseline [Na]_in = {Na_inside_baseline:.0f} mM')

    ax4.fill_between(t, Na_inside_baseline, Na_in,
                     where=Na_in > Na_inside_baseline, interpolate=True,
                     color='darkorange', alpha=0.25,
                     label='Na⁺ accumulation above baseline')

    ax4.set_ylabel('[Na]$_{in}$\n(mM)', fontsize=10)
    ax4.set_xlabel('Time (ms)', fontsize=10)
    ax4.set_title('Intracellular Sodium — Small Trickle vs Large Flood', fontsize=11)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_open_fraction_closeup(results, pulse_times_ms, pulse_amplitudes, pulse_duration_ms):
    """
    Overlay the sodium channel open fraction response to each pulse,
    time-aligned to the pulse onset. Clearly shows how sub-threshold pulses
    produce a small, transient channel opening while the supra-threshold
    pulse triggers a massive, self-amplifying opening.
    """
    t         = results['t']
    open_frac = results['open_frac']
    V         = results['V']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7))
    fig.suptitle('Sub- vs Supra-Threshold Response\n(aligned to each pulse onset)',
                 fontsize=13, fontweight='bold')

    window_ms = 60.0   # ms to show after each pulse onset
    dt = t[1] - t[0]
    win_steps = int(window_ms / dt)

    colors  = PULSE_COLORS[:len(pulse_times_ms)]

    legend_patches = []
    for t_on, amp, color in zip(pulse_times_ms, pulse_amplitudes, colors):
        idx_on = np.searchsorted(t, t_on)
        idx_end = min(idx_on + win_steps, len(t))
        t_rel = t[idx_on:idx_end] - t_on

        v_seg   = V[idx_on:idx_end]
        of_seg  = open_frac[idx_on:idx_end]
        fired   = np.any(v_seg > 10.0)

        lw    = 2.2 if fired else 1.4
        ls    = '-'
        label = f'{amp:.0f} µA/cm²' + (' — ACTION POTENTIAL' if fired else ' — subthreshold')

        ax_top.plot(t_rel, v_seg,  color=color, linewidth=lw, linestyle=ls)
        ax_bot.plot(t_rel, of_seg, color=color, linewidth=lw, linestyle=ls)

        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax_top.axhline(V_rest, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax_top.axhline(10.0,   color='tomato', linestyle='--', linewidth=1,
                   alpha=0.7, label='AP criterion (10 mV)')
    ax_top.set_ylabel('Membrane Potential (mV)', fontsize=10)
    ax_top.set_title('Voltage Response', fontsize=11)
    ax_top.set_ylim([-75, 55])
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(handles=legend_patches + [
        mpatches.Patch(color='tomato', label='AP criterion (10 mV)')
    ], fontsize=8.5, loc='upper right')

    m0, h0, _ = steady_state(V_rest)
    ax_bot.axhline(m0**3 * h0, color='gray', linestyle=':', linewidth=1,
                   alpha=0.6, label='Resting open fraction')
    ax_bot.set_ylabel('Na channel open fraction\n(m³ · h)', fontsize=10)
    ax_bot.set_xlabel('Time after pulse onset (ms)', fontsize=10)
    ax_bot.set_title('Sodium Channel Activation', fontsize=11)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(handles=legend_patches, fontsize=8.5, loc='upper right')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("SODIUM CHANNEL ACTIVATION AND THE ACTION POTENTIAL THRESHOLD")
    print("=" * 65)

    # Define 5 pulses of increasing strength
    # The threshold for the HH model with 2 ms pulses is ~9-11 µA/cm²
    pulse_times_ms   = [30.0, 80.0, 130.0, 185.0, 245.0]
    pulse_amplitudes = [3.0,  6.0,  9.0,   11.0,  14.0]   # µA/cm²
    pulse_duration_ms = 2.0

    print(f"""
Simulation setup
----------------
Pulse duration : {pulse_duration_ms} ms each
Pulse strengths: {pulse_amplitudes} µA/cm²
Pulse onset times: {pulse_times_ms} ms

The neuron starts at rest ({V_rest} mV). Each pulse briefly injects
depolarizing current. We track three things:

  1. Membrane voltage
  2. Fraction of open sodium channels (m³ · h)
  3. Intracellular [Na⁺] — rising when channels open, slowly
     restored between pulses by the Na/K pump (τ ≈ {tau_pump_ms:.0f} ms)
""")

    print("Running simulation...")
    results = run_threshold_simulation(
        pulse_times_ms=pulse_times_ms,
        pulse_amplitudes=pulse_amplitudes,
        pulse_duration_ms=pulse_duration_ms,
    )

    # ── Summarise each pulse ──────────────────────────────────────────────
    t  = results['t']
    V  = results['V']
    Na = results['Na_in']

    print("\nPulse-by-pulse summary")
    print("-" * 55)
    print(f"{'Pulse':>5}  {'Amp (µA/cm²)':>13}  {'Peak V (mV)':>11}  {'ΔNa (µM)':>9}  {'Outcome':>14}")
    print("-" * 55)

    for k, (t_on, amp) in enumerate(zip(pulse_times_ms, pulse_amplitudes)):
        window = (t >= t_on) & (t < t_on + 55.0)
        peak_V = V[window].max()
        Na_baseline = Na[np.searchsorted(t, t_on)]
        delta_Na_uM = (Na[window].max() - Na_baseline) * 1000.0  # mM → µM
        outcome = "ACTION POTENTIAL" if peak_V > 10.0 else "subthreshold"
        print(f"  #{k+1}    {amp:>10.1f}     {peak_V:>10.1f}   {delta_Na_uM:>8.2f}   {outcome}")

    print("-" * 55)
    print(f"""
Interpretation
--------------
Subthreshold pulses partially open sodium channels. Na+ enters the
cell and the membrane depolarizes — but not enough to sustain itself.
The inactivation gate (h) closes, leak current pulls the voltage back
to rest, and [Na]_in rises by only a few µM.

Once the stimulus crosses threshold, the initial depolarization opens
enough channels that the resulting Na+ influx depolarizes the membrane
FURTHER, opening even more channels. This positive feedback loop is
the action potential. [Na]_in rises by ~0.1 mM — roughly 100× the
change from a subthreshold pulse.

The h gate (inactivation) eventually cuts off the Na+ flood, and K+
channels repolarize the membrane — completing the spike.
""")

    print("Generating plots...")
    plot_threshold_experiment(results, pulse_times_ms, pulse_amplitudes, pulse_duration_ms)
    plot_open_fraction_closeup(results, pulse_times_ms, pulse_amplitudes, pulse_duration_ms)


if __name__ == "__main__":
    main()
