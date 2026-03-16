"""
Stochastic Hodgkin-Huxley Neuron Simulation
===========================================

This module demonstrates how neurons process information using:
- Discrete ion channels with stochastic gating
- Random synaptic input (excitatory events)

The Hodgkin-Huxley model describes how action potentials are initiated
and propagated in neurons through the interplay of voltage-gated ion
channels (primarily sodium and potassium).

In this stochastic version, we model individual ion channels that open
and close probabilistically, creating realistic "channel noise" that
affects neuronal dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# BIOLOGICAL PARAMETERS
# =============================================================================

# Membrane capacitance (uF/cm^2)
# This determines how quickly the membrane voltage can change
C_m = 1.0

# Maximum conductances (mS/cm^2)
# These set the maximum current each channel type can carry
g_Na_max = 120.0  # Sodium - responsible for rapid depolarization
g_K_max = 36.0    # Potassium - responsible for repolarization
g_L = 0.3         # Leak - maintains resting potential

# Reversal potentials (mV)
# These are the voltages at which each ion type has zero net flow
E_Na = 50.0       # Sodium reversal potential (positive, causes depolarization)
E_K = -77.0       # Potassium reversal potential (negative, causes repolarization)
E_L = -54.387     # Leak reversal potential (near resting potential)

# Number of ion channels for stochastic model
# More channels = less noise, fewer channels = more noise
N_Na = 6000       # Number of sodium channels (typical: 1000-10000)
N_K = 1800        # Number of potassium channels (typical: 500-5000)

# Single channel conductances (calculated from max conductance / N channels)
gamma_Na = g_Na_max / N_Na  # Single Na channel conductance
gamma_K = g_K_max / N_K     # Single K channel conductance

# Synaptic parameters
E_syn = 0.0       # Excitatory synapse reversal potential (mV)
g_syn_max = 0.5   # Maximum synaptic conductance (mS/cm^2)
tau_syn = 5.0     # Synaptic decay time constant (ms)


# =============================================================================
# GATING VARIABLE FUNCTIONS
# =============================================================================

def alpha_m(V):
    """
    Opening rate for sodium activation gate (m).

    The m gate is fast and controls the initial sodium influx
    during an action potential.
    """
    # Avoid division by zero
    dV = V + 40.0
    if np.isscalar(dV):
        if np.abs(dV) < 1e-7:
            return 1.0
        return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))
    result = np.zeros_like(dV)
    mask = np.abs(dV) < 1e-7
    result[mask] = 1.0
    result[~mask] = 0.1 * dV[~mask] / (1.0 - np.exp(-dV[~mask] / 10.0))
    return result


def beta_m(V):
    """Closing rate for sodium activation gate (m)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    """
    Opening rate for sodium inactivation gate (h).

    The h gate is slower and automatically closes after sodium
    channels open, preventing sustained sodium current.
    """
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    """Closing rate for sodium inactivation gate (h)."""
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n(V):
    """
    Opening rate for potassium activation gate (n).

    The n gate controls potassium efflux, which repolarizes
    the membrane after sodium influx.
    """
    dV = V + 55.0
    if np.isscalar(dV):
        if np.abs(dV) < 1e-7:
            return 0.1
        return 0.01 * dV / (1.0 - np.exp(-dV / 10.0))
    result = np.zeros_like(dV)
    mask = np.abs(dV) < 1e-7
    result[mask] = 0.1
    result[~mask] = 0.01 * dV[~mask] / (1.0 - np.exp(-dV[~mask] / 10.0))
    return result


def beta_n(V):
    """Closing rate for potassium activation gate (n)."""
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def steady_state(V):
    """
    Calculate steady-state values for all gating variables.

    At any voltage, channels will tend toward these values
    given enough time.

    Returns: (m_inf, h_inf, n_inf)
    """
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)

    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)

    return m_inf, h_inf, n_inf


def time_constants(V):
    """
    Calculate time constants for all gating variables.

    These determine how quickly each gate responds to voltage changes.

    Returns: (tau_m, tau_h, tau_n) in ms
    """
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)

    tau_m = 1.0 / (am + bm)
    tau_h = 1.0 / (ah + bh)
    tau_n = 1.0 / (an + bn)

    return tau_m, tau_h, tau_n


# =============================================================================
# STOCHASTIC CHANNEL MODEL
# =============================================================================

class StochasticChannels:
    """
    Stochastic ion channel population using a binomial/Markov approach.

    Real neurons have discrete ion channels that open and close randomly.
    This class tracks the fraction of open channels, updating them
    probabilistically at each timestep.

    The number of open channels follows a binomial distribution, creating
    "channel noise" - small random fluctuations in membrane potential
    that can trigger spontaneous spikes near threshold.
    """

    def __init__(self, N_Na=N_Na, N_K=N_K, V_init=-65.0):
        """
        Initialize channel populations.

        Parameters
        ----------
        N_Na : int
            Number of sodium channels
        N_K : int
            Number of potassium channels
        V_init : float
            Initial membrane voltage for setting initial gate states
        """
        self.N_Na = N_Na
        self.N_K = N_K

        # Initialize gating variables to steady-state values
        m_inf, h_inf, n_inf = steady_state(V_init)
        self.m = m_inf
        self.h = h_inf
        self.n = n_inf

        # For sodium channels: fraction open = m^3 * h
        # We track m and h separately for proper dynamics
        # Number of open Na channels
        self.n_Na_open = int(N_Na * (m_inf ** 3) * h_inf)

        # Number of open K channels (n^4)
        self.n_K_open = int(N_K * (n_inf ** 4))

    def update(self, V, dt):
        """
        Update channel states stochastically.

        Uses the two-state Markov model for each gate:
        - Probability of opening: alpha * dt
        - Probability of closing: beta * dt

        Parameters
        ----------
        V : float
            Current membrane voltage (mV)
        dt : float
            Time step (ms)
        """
        # Get rate constants at current voltage
        am, bm = alpha_m(V), beta_m(V)
        ah, bh = alpha_h(V), beta_h(V)
        an, bn = alpha_n(V), beta_n(V)

        # Update m gate (sodium activation)
        # Number of closed m gates that could open
        n_m_closed = int(self.N_Na * (1 - self.m))
        # Number of open m gates that could close
        n_m_open = int(self.N_Na * self.m)

        # Stochastic transitions
        p_open_m = am * dt  # Probability of opening per timestep
        p_close_m = bm * dt  # Probability of closing per timestep

        # Clamp probabilities to [0, 1]
        p_open_m = np.clip(p_open_m, 0, 1)
        p_close_m = np.clip(p_close_m, 0, 1)

        # Number that actually transition
        if n_m_closed > 0:
            opening_m = np.random.binomial(n_m_closed, p_open_m)
        else:
            opening_m = 0
        if n_m_open > 0:
            closing_m = np.random.binomial(n_m_open, p_close_m)
        else:
            closing_m = 0

        # Update m fraction
        self.m += (opening_m - closing_m) / self.N_Na
        self.m = np.clip(self.m, 0, 1)

        # Update h gate (sodium inactivation) - similar process
        n_h_closed = int(self.N_Na * (1 - self.h))
        n_h_open = int(self.N_Na * self.h)

        p_open_h = np.clip(ah * dt, 0, 1)
        p_close_h = np.clip(bh * dt, 0, 1)

        if n_h_closed > 0:
            opening_h = np.random.binomial(n_h_closed, p_open_h)
        else:
            opening_h = 0
        if n_h_open > 0:
            closing_h = np.random.binomial(n_h_open, p_close_h)
        else:
            closing_h = 0

        self.h += (opening_h - closing_h) / self.N_Na
        self.h = np.clip(self.h, 0, 1)

        # Update n gate (potassium activation)
        n_n_closed = int(self.N_K * (1 - self.n))
        n_n_open = int(self.N_K * self.n)

        p_open_n = np.clip(an * dt, 0, 1)
        p_close_n = np.clip(bn * dt, 0, 1)

        if n_n_closed > 0:
            opening_n = np.random.binomial(n_n_closed, p_open_n)
        else:
            opening_n = 0
        if n_n_open > 0:
            closing_n = np.random.binomial(n_n_open, p_close_n)
        else:
            closing_n = 0

        self.n += (opening_n - closing_n) / self.N_K
        self.n = np.clip(self.n, 0, 1)

        # Calculate effective open channel fractions
        # Na channels need m^3 * h to be conducting
        self.n_Na_open = self.N_Na * (self.m ** 3) * self.h
        # K channels need n^4 to be conducting
        self.n_K_open = self.N_K * (self.n ** 4)

    def get_conductances(self):
        """
        Get current conductances from open channel fractions.

        Returns
        -------
        g_Na : float
            Sodium conductance (mS/cm^2)
        g_K : float
            Potassium conductance (mS/cm^2)
        """
        g_Na = gamma_Na * self.n_Na_open
        g_K = gamma_K * self.n_K_open
        return g_Na, g_K


# =============================================================================
# SYNAPTIC INPUT MODEL
# =============================================================================

def generate_synaptic_events(duration, rate, dt):
    """
    Generate random synaptic input events using a Poisson process.

    In real neurons, synaptic inputs arrive at unpredictable times
    due to the random nature of presynaptic activity.

    Parameters
    ----------
    duration : float
        Total simulation duration (ms)
    rate : float
        Average rate of synaptic events (events per ms, i.e., kHz)
    dt : float
        Time step (ms)

    Returns
    -------
    events : ndarray
        Boolean array indicating when synaptic events occur
    """
    n_steps = int(duration / dt)
    # Probability of event in each timestep
    p_event = rate * dt
    # Generate random events
    events = np.random.random(n_steps) < p_event
    return events


def update_synaptic_conductance(g_syn, event, dt):
    """
    Update synaptic conductance with decay and new events.

    Synaptic conductance rises instantly when an event occurs,
    then decays exponentially with time constant tau_syn.

    Parameters
    ----------
    g_syn : float
        Current synaptic conductance (mS/cm^2)
    event : bool
        Whether a synaptic event occurred this timestep
    dt : float
        Time step (ms)

    Returns
    -------
    g_syn : float
        Updated synaptic conductance
    """
    # Exponential decay
    g_syn *= np.exp(-dt / tau_syn)

    # Add new event
    if event:
        g_syn += g_syn_max

    return g_syn


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(duration=500.0, dt=0.01, synaptic_rate=0.05,
                   stochastic=True, seed=None):
    """
    Run the Hodgkin-Huxley neuron simulation.

    The simulation uses Euler integration to solve the membrane
    voltage equation:

        C_m * dV/dt = I_syn - I_Na - I_K - I_L

    where:
        I_Na = g_Na * (V - E_Na)  [sodium current]
        I_K  = g_K * (V - E_K)    [potassium current]
        I_L  = g_L * (V - E_L)    [leak current]
        I_syn = g_syn * (V - E_syn)  [synaptic current]

    Parameters
    ----------
    duration : float
        Simulation duration (ms)
    dt : float
        Time step (ms)
    synaptic_rate : float
        Rate of synaptic events (events per ms)
    stochastic : bool
        Whether to use stochastic channel model
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    results : dict
        Dictionary containing simulation results
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize time array
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)

    # Initialize state variables
    V = np.zeros(n_steps)
    V[0] = -65.0  # Resting potential (mV)

    # Initialize channel model
    if stochastic:
        channels = StochasticChannels(V_init=V[0])
    else:
        # Deterministic version uses continuous gating variables
        m_inf, h_inf, n_inf = steady_state(V[0])
        m, h, n = m_inf, h_inf, n_inf

    # Arrays to store channel states
    m_trace = np.zeros(n_steps)
    h_trace = np.zeros(n_steps)
    n_trace = np.zeros(n_steps)

    # Arrays to store currents
    I_Na_trace = np.zeros(n_steps)
    I_K_trace = np.zeros(n_steps)
    I_L_trace = np.zeros(n_steps)
    I_syn_trace = np.zeros(n_steps)

    # Generate synaptic events
    synaptic_events = generate_synaptic_events(duration, synaptic_rate, dt)
    g_syn = 0.0

    # Arrays for conductances
    g_Na_trace = np.zeros(n_steps)
    g_K_trace = np.zeros(n_steps)
    g_syn_trace = np.zeros(n_steps)

    # Main simulation loop
    print("\nRunning simulation...")
    print(f"  Duration: {duration} ms")
    print(f"  Time step: {dt} ms")
    print(f"  Stochastic channels: {stochastic}")
    print(f"  Synaptic input rate: {synaptic_rate * 1000:.1f} Hz")

    for i in range(n_steps - 1):
        # Update synaptic conductance
        g_syn = update_synaptic_conductance(g_syn, synaptic_events[i], dt)
        g_syn_trace[i] = g_syn

        # Get ion channel conductances
        if stochastic:
            g_Na, g_K = channels.get_conductances()
            m_trace[i] = channels.m
            h_trace[i] = channels.h
            n_trace[i] = channels.n
        else:
            g_Na = g_Na_max * (m ** 3) * h
            g_K = g_K_max * (n ** 4)
            m_trace[i] = m
            h_trace[i] = h
            n_trace[i] = n

        g_Na_trace[i] = g_Na
        g_K_trace[i] = g_K

        # Calculate currents
        I_Na = g_Na * (V[i] - E_Na)
        I_K = g_K * (V[i] - E_K)
        I_L = g_L * (V[i] - E_L)
        I_syn = g_syn * (V[i] - E_syn)

        I_Na_trace[i] = I_Na
        I_K_trace[i] = I_K
        I_L_trace[i] = I_L
        I_syn_trace[i] = I_syn

        # Euler integration of membrane voltage
        # dV/dt = (1/C_m) * (-I_Na - I_K - I_L + I_syn)
        # Note: I_syn is excitatory (drives V toward E_syn=0)
        # Ion currents are defined as outward positive, so we subtract them
        dV = (1.0 / C_m) * (-I_Na - I_K - I_L - I_syn) * dt
        V[i + 1] = V[i] + dV

        # Update channel states
        if stochastic:
            channels.update(V[i + 1], dt)
        else:
            # Deterministic update using rate equations
            am, bm = alpha_m(V[i + 1]), beta_m(V[i + 1])
            ah, bh = alpha_h(V[i + 1]), beta_h(V[i + 1])
            an, bn = alpha_n(V[i + 1]), beta_n(V[i + 1])

            m += (am * (1 - m) - bm * m) * dt
            h += (ah * (1 - h) - bh * h) * dt
            n += (an * (1 - n) - bn * n) * dt

    # Fill in last values
    g_syn_trace[-1] = g_syn
    if stochastic:
        m_trace[-1] = channels.m
        h_trace[-1] = channels.h
        n_trace[-1] = channels.n
        g_Na_trace[-1], g_K_trace[-1] = channels.get_conductances()
    else:
        m_trace[-1] = m
        h_trace[-1] = h
        n_trace[-1] = n
        g_Na_trace[-1] = g_Na_max * (m ** 3) * h
        g_K_trace[-1] = g_K_max * (n ** 4)

    I_Na_trace[-1] = g_Na_trace[-1] * (V[-1] - E_Na)
    I_K_trace[-1] = g_K_trace[-1] * (V[-1] - E_K)
    I_L_trace[-1] = g_L * (V[-1] - E_L)
    I_syn_trace[-1] = g_syn_trace[-1] * (V[-1] - E_syn)

    # Count spikes (threshold crossings from below)
    threshold = 0.0  # mV
    spikes = np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    print(f"\n  Detected {len(spikes)} action potentials")

    results = {
        't': t,
        'V': V,
        'm': m_trace,
        'h': h_trace,
        'n': n_trace,
        'g_Na': g_Na_trace,
        'g_K': g_K_trace,
        'g_syn': g_syn_trace,
        'I_Na': I_Na_trace,
        'I_K': I_K_trace,
        'I_L': I_L_trace,
        'I_syn': I_syn_trace,
        'synaptic_events': synaptic_events,
        'spikes': spikes,
        'dt': dt
    }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results, save_path=None):
    """
    Create a multi-panel figure showing simulation results.

    Panel 1: Membrane potential over time (action potentials)
    Panel 2: Synaptic input events and conductance
    Panel 3: Ion channel states (gating variables)
    Panel 4: Ion currents

    Parameters
    ----------
    results : dict
        Dictionary of simulation results from run_simulation()
    save_path : str, optional
        Path to save the figure
    """
    t = results['t']

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Stochastic Hodgkin-Huxley Neuron Simulation',
                 fontsize=14, fontweight='bold')

    # Panel 1: Membrane potential
    ax1 = axes[0]
    ax1.plot(t, results['V'], 'k-', linewidth=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_ylabel('Membrane\nPotential (mV)')
    ax1.set_ylim([-90, 60])
    ax1.legend(loc='upper right')
    ax1.set_title('Action Potentials Generated by Random Synaptic Input')

    # Mark detected spikes
    if len(results['spikes']) > 0:
        spike_times = t[results['spikes']]
        ax1.scatter(spike_times, [50] * len(spike_times), color='red',
                   marker='v', s=30, label='Spike', zorder=5)

    # Panel 2: Synaptic input
    ax2 = axes[1]

    # Plot synaptic events as a raster
    event_times = t[results['synaptic_events']]
    ax2.eventplot([event_times], colors='blue', lineoffsets=0.8,
                  linelengths=0.3, linewidths=0.5)

    # Plot synaptic conductance on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(t, 0, results['g_syn'], alpha=0.3, color='blue')
    ax2_twin.set_ylabel('g_syn (mS/cm²)', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')
    ax2_twin.set_ylim([0, max(results['g_syn']) * 1.2 + 0.1])

    ax2.set_ylabel('Synaptic\nEvents')
    ax2.set_yticks([])
    ax2.set_title('Excitatory Synaptic Input (Poisson Process)')

    # Panel 3: Gating variables
    ax3 = axes[2]
    ax3.plot(t, results['m'], 'r-', linewidth=0.8, label='m (Na activation)', alpha=0.8)
    ax3.plot(t, results['h'], 'orange', linewidth=0.8, label='h (Na inactivation)', alpha=0.8)
    ax3.plot(t, results['n'], 'b-', linewidth=0.8, label='n (K activation)', alpha=0.8)
    ax3.set_ylabel('Gating\nVariables')
    ax3.set_ylim([0, 1])
    ax3.legend(loc='upper right', ncol=3, fontsize=9)
    ax3.set_title('Ion Channel Gating Variables (Stochastic Dynamics)')

    # Panel 4: Ion currents
    ax4 = axes[3]
    ax4.plot(t, -results['I_Na'], 'r-', linewidth=0.8, label='I_Na (inward)', alpha=0.8)
    ax4.plot(t, -results['I_K'], 'b-', linewidth=0.8, label='I_K (outward)', alpha=0.8)
    ax4.plot(t, -results['I_syn'], 'g-', linewidth=0.8, label='I_syn', alpha=0.8)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Current\n(µA/cm²)')
    ax4.set_xlabel('Time (ms)')
    ax4.legend(loc='upper right', ncol=3, fontsize=9)
    ax4.set_title('Ionic Currents (Positive = Inward/Depolarizing)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def plot_comparison(stochastic_results, deterministic_results, save_path=None):
    """
    Compare stochastic and deterministic simulations side by side.

    Parameters
    ----------
    stochastic_results : dict
        Results from stochastic simulation
    deterministic_results : dict
        Results from deterministic simulation
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Stochastic vs Deterministic Hodgkin-Huxley Model',
                 fontsize=14, fontweight='bold')

    t = stochastic_results['t']

    # Stochastic
    axes[0].plot(t, stochastic_results['V'], 'k-', linewidth=0.8)
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title(f'Stochastic ({N_Na} Na channels, {N_K} K channels) - '
                      f'{len(stochastic_results["spikes"])} spikes')
    axes[0].set_ylim([-90, 60])

    # Deterministic
    axes[1].plot(t, deterministic_results['V'], 'k-', linewidth=0.8)
    axes[1].set_ylabel('V (mV)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title(f'Deterministic (Continuous gating variables) - '
                      f'{len(deterministic_results["spikes"])} spikes')
    axes[1].set_ylim([-90, 60])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Run the main demonstration of the stochastic Hodgkin-Huxley neuron.
    """
    print("=" * 70)
    print("STOCHASTIC HODGKIN-HUXLEY NEURON SIMULATION")
    print("=" * 70)

    print("\n--- INTRODUCTION ---")
    print("""
This simulation demonstrates how a neuron processes incoming information:

1. SYNAPTIC INPUT: Random excitatory events arrive at the neuron
   (like signals from other neurons). These are modeled as a Poisson
   process - unpredictable but with a defined average rate.

2. ION CHANNELS: The neuron has thousands of sodium (Na+) and potassium (K+)
   channels. Unlike textbook models, real channels open and close randomly,
   creating "channel noise" that affects neural dynamics.

3. ACTION POTENTIALS: When enough synaptic input accumulates, the membrane
   voltage crosses threshold, triggering an action potential - the neuron
   "fires" and sends a signal to downstream neurons.
""")

    print("\n--- RUNNING STOCHASTIC SIMULATION ---")
    # Run stochastic simulation with a fixed seed for reproducibility
    stochastic_results = run_simulation(
        duration=200.0,
        dt=0.01,
        synaptic_rate=0.05,  # 50 Hz average input rate
        stochastic=True,
        seed=42
    )

    print("\n--- RUNNING DETERMINISTIC SIMULATION (for comparison) ---")
    # Run deterministic simulation with same synaptic input
    np.random.seed(42)  # Same seed for same synaptic events
    deterministic_results = run_simulation(
        duration=200.0,
        dt=0.01,
        synaptic_rate=0.05,
        stochastic=False,
        seed=42
    )

    print("\n--- KEY OBSERVATIONS ---")
    print(f"""
Results Summary:
- Stochastic model: {len(stochastic_results['spikes'])} action potentials
- Deterministic model: {len(deterministic_results['spikes'])} action potentials

Key differences you may observe:
1. CHANNEL NOISE: The stochastic model shows small voltage fluctuations
   even at rest due to random channel opening/closing.

2. SPIKE TIMING: Due to channel noise, the stochastic model may fire
   at slightly different times than the deterministic model.

3. THRESHOLD VARIABILITY: Channel noise can push voltage across threshold
   earlier or later than expected, affecting spike timing.

In biological neurons, this stochasticity is thought to play important
roles in information processing and neural computation.
""")

    print("\n--- GENERATING VISUALIZATIONS ---")
    print("Creating detailed results plot...")
    plot_results(stochastic_results)

    print("\nCreating comparison plot...")
    plot_comparison(stochastic_results, deterministic_results)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return stochastic_results, deterministic_results


if __name__ == "__main__":
    main()
