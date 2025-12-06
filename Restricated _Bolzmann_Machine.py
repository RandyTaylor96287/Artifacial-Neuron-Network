import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import expit # Efficient logistic(x) = 1 / (1 + exp(-x))

# --- 1. Define XOR Dataset ---
DATA_XOR = np.array([
    [-1, -1, -1],
    [-1,  1,  1],
    [ 1, -1,  1],
    [ 1,  1, -1]
], dtype=np.float32)

# Generate all 8 (2^3) possible v patterns
ALL_PATTERNS = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float32)

# Define P_true (target distribution)
P_TRUE = np.zeros(8)
xor_indices = [
    0, # [-1, -1, -1]
    3, # [-1,  1,  1]
    5, # [ 1, -1,  1]
    6  # [ 1,  1, -1]
]
P_TRUE[xor_indices] = 0.25


# --- 2. RBM (Spin +/-1) Class ---

class RBM:
    """
    RBM implementation for +/-1 (spin) neurons.
    Uses the correct formulas for Algorithm 3.
    """
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize weights and biases
        # Use fixed random seed for reproducibility
        rng = np.random.default_rng(1234)
        self.W = rng.normal(0, 0.1, size=(n_visible, n_hidden))
        self.b_v = np.zeros(n_visible) # visible layer bias a
        self.b_h = np.zeros(n_hidden) # hidden layer bias b

    # --- Core probability functions (for +/-1) ---

    def _logistic(self, x):
        """ P(s=+1) = logistic(2 * I) """
        return expit(2 * x)

    def _tanh(self, x):
        """ <S> = tanh(I) """
        return np.tanh(x)

    def _sample_h_given_v(self, v):
        """ Sample h from P(h|v) """
        v_flat = v.reshape(-1, self.n_visible)
        h_input = v_flat @ self.W + self.b_h
        
        prob_h_pos = self._logistic(h_input)
        
        # Sample +/-1
        h_sample = np.where(np.random.rand(*prob_h_pos.shape) < prob_h_pos, 1, -1)
        return h_sample

    def _sample_v_given_h(self, h):
        """ Sample v from P(v|h) """
        h_flat = h.reshape(-1, self.n_hidden)
        v_input = h_flat @ self.W.T + self.b_v
        
        prob_v_pos = self._logistic(v_input)
        
        # Sample +/-1
        v_sample = np.where(np.random.rand(*prob_v_pos.shape) < prob_v_pos, 1, -1)
        return v_sample

    def _mean_h_given_v(self, v):
        """ Calculate expectation <h_j|v> """
        v_flat = v.reshape(-1, self.n_visible)
        h_input = v_flat @ self.W + self.b_h
        return self._tanh(h_input) # <h_j> = tanh(I_j)

    # --- Training and Evaluation ---

    def train(self, data, epochs, k, eta):
        """ Train RBM using CD-k algorithm """
        n_samples = data.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data (minor impact for batch size 4)
            idx = np.random.permutation(n_samples)
            data_shuffled = data[idx]
            
            # --- Full CD-k ---
            # (Using full dataset as one mini-batch)
            
            # --- 1. Positive Phase (data-driven) ---
            v_0 = data_shuffled
            mean_h_0 = self._mean_h_given_v(v_0) # <h>_data = tanh(I_0)
            
            # --- 2. Negative Phase (model-driven) ---
            # Start k-step Gibbs chain from v_0
            v_k = v_0
            for _ in range(k):
                h_k = self._sample_h_given_v(v_k)
                v_k = self._sample_v_given_h(h_k)
            
            # Calculate expectation at v_k
            mean_h_k = self._mean_h_given_v(v_k) # <h>_model = tanh(I_k)
            
            # --- 3. Parameter Update ---
            self.W += eta * (v_0.T @ mean_h_0 - v_k.T @ mean_h_k) / n_samples
            self.b_v += eta * np.mean(v_0 - v_k, axis=0)
            self.b_h += eta * np.mean(mean_h_0 - mean_h_k, axis=0)
            
            if (epoch + 1) % 10000 == 0:
                print(f"  Epoch {epoch+1}/{epochs} completed (M={self.n_hidden})")

    def get_exact_probabilities(self, all_patterns):
        """
        Analytically calculate P_model(v) for all 8 patterns via free energy.
        """
        free_energies = np.zeros(all_patterns.shape[0])
        
        for i, v in enumerate(all_patterns):
            # F(v) = -a^T v - sum_j log(2*cosh(b_j + v^T W_j))
            visible_term = -np.dot(self.b_v, v)
            h_input = self.b_h + v @ self.W
            hidden_term = -np.sum(np.log(2 * np.cosh(h_input)))
            free_energies[i] = visible_term + hidden_term
            
        # P_model(v) = exp(-F(v)) / Z
        # Z = sum_v' exp(-F(v'))
        
        # Subtract F_min for numerical stability
        f_min = np.min(free_energies)
        unnormalized_probs = np.exp(-(free_energies - f_min))
        
        Z = np.sum(unnormalized_probs)
        P_model = unnormalized_probs / Z
        
        return P_model

    def run_gibbs_sampling(self, all_patterns, n_steps, burn_in=10000):
        """
        Estimate P_model(v) via long Gibbs sampling run.
        (Used to answer the "run dynamics" question)
        """
        print(f"  ... Running Gibbs sampling {n_steps} steps ...")
        pattern_map = {tuple(p): i for i, p in enumerate(all_patterns)} # dict keys
        counts = np.zeros(all_patterns.shape[0])
        
        # Start from a random pattern
        v = all_patterns[np.random.randint(0, 8)]
        
        for i in range(n_steps + burn_in):
            h = self._sample_h_given_v(v)
            v_sample = self._sample_v_given_h(h)
            v = v_sample[0] # v_sample is (1, N_v) array
            
            if i >= burn_in:
                counts[pattern_map[tuple(v)]] += 1 # Record this v
                
        P_gibbs = counts / n_steps # Convert counts to probabilities
        return P_gibbs

# --- 3. Helper Functions ---

def calculate_kl_divergence(P_true, Q_model):
    """
    Calculate KL Divergence D_KL(P || Q)
    D_KL = sum P(v) * log2( P(v) / Q(v) )
    """
    epsilon = 1e-9 # Add epsilon to avoid log(0)
    Q_model_safe = np.clip(Q_model, epsilon, 1.0)
    
    # Sum only over P_true > 0 terms (the 4 XOR patterns)
    non_zero_indices = np.where(P_TRUE > 0)[0]
    
    P_non_zero = P_true[non_zero_indices]
    Q_non_zero = Q_model_safe[non_zero_indices]
    
    kl_div = np.sum(P_non_zero * (np.log2(P_non_zero) - np.log2(Q_non_zero)))
    return kl_div

# --- 4. Main Experiment Script ---

if __name__ == "__main__":
    
    # --- Experiment Parameters ---
    M_H_LIST = [1, 2, 3, 4, 8] # Number of hidden units
    KL_THEORY = [1.0, 0.278, 0.0, 0.0, 0.0] # Theoretical minimum KL divergence (bits)
    
    # --- Training Hyperparameters ---
    EPOCHS = 40000       # Training epochs
    LEARNING_RATE = 0.015 # Learning rate (eta)
    K_CD = 10            # k for CD-k
    
    print("Starting RBM Experiment...")
    print(f"Hyperparameters: Epochs={EPOCHS}, eta={LEARNING_RATE}, k={K_CD}\n")
    
    experimental_kls = []
    trained_rbms = {}

    # --- Main Loop: Training ---
    for M in M_H_LIST:
        print(f"--- Training RBM M={M} ---")
        rbm = RBM(n_visible=3, n_hidden=M)
        rbm.train(DATA_XOR, epochs=EPOCHS, k=K_CD, eta=LEARNING_RATE)
        
        # After training, calculate exact P_model(v)
        Q_rbm = rbm.get_exact_probabilities(ALL_PATTERNS)
        
        print("\n  P_true vs P_model:")
        for i, p in enumerate(ALL_PATTERNS):
            print(f"    v={p}: P_true={P_TRUE[i]:.3f}, P_model={Q_rbm[i]:.3f}")
            
        # Calculate KL Divergence
        kl = calculate_kl_divergence(P_TRUE, Q_rbm)
        experimental_kls.append(kl)
        trained_rbms[M] = rbm # Save for later use
        print(f"  KL Divergence D_KL(P_true || P_model) = {kl:.4f} bits\n")

    # --- Plotting ---
    print("--- Plotting Results ---")
    
    # --- Plot 1: KL Divergence vs M ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(M_H_LIST, experimental_kls, 'bo-', label='Experimental KL (CD-k)', markersize=10)
    
    theory_M_values = [1, 2, 3, 4, 8]
    plt.plot(theory_M_values, KL_THEORY, 'r--', label='Theoretical Minimum KL', markersize=10)
    
    plt.title(f'RBM learn XOR (N=3) (k={K_CD}, $\eta$={LEARNING_RATE}, epochs={EPOCHS})')
    plt.xlabel("Number of Hidden Units (M)")
    plt.ylabel("KL Divergence (bits)")
    plt.xticks(theory_M_values) # Ensure M=3 is on the x-axis ticks
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=-0.1) # Allow y=0 to be visible
    
    plt.savefig("rbm_xor_kl_divergence.png")
    
    # --- [ Start of new plot block ] ---
    # Plot 2: Probability Distribution Comparison (to "count the frequencies")
    print("\n--- Plotting Probability Distributions ---")
    fig_probs, axes = plt.subplots(len(M_H_LIST), 1, figsize=(10, 15), sharey=True)
    fig_probs.suptitle('P_true vs. P_model(v) for different M', fontsize=16)
    
    bar_width = 0.35
    index = np.arange(8)
    pattern_labels = [str(tuple(int(x) for x in p)) for p in ALL_PATTERNS]

    for i, M in enumerate(M_H_LIST):
        rbm = trained_rbms[M]
        Q_model = rbm.get_exact_probabilities(ALL_PATTERNS)
        
        ax = axes[i] if len(M_H_LIST) > 1 else axes # Handle single subplot case
        
        ax.bar(index - bar_width/2, P_TRUE, bar_width, label='P_true (XOR)', color='red', alpha=0.9)
        ax.bar(index + bar_width/2, Q_model, bar_width, label=f'P_model (M={M})', color='blue', alpha=0.7)
        
        ax.set_title(f'M = {M} (KL = {experimental_kls[i]:.4f} bits)')
        ax.set_ylabel('Probability')
        ax.set_xticks(index)
        ax.set_xticklabels(pattern_labels, rotation=30, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', axis='y', alpha=0.6)

    axes[0].set_ylim(0, 0.6) # Set uniform Y-axis limit
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
    plt.savefig("rbm_xor_probabilities.png")
    print("Probability distribution plot saved to 'rbm_xor_probabilities.png'\n")
    # --- [ End of new plot block ] ---
    
    rbm_m4 = trained_rbms[4] 
    Q_exact = rbm_m4.get_exact_probabilities(ALL_PATTERNS)
    KL_exact = calculate_kl_divergence(P_TRUE, Q_exact)
    
    print(f"  M=4 Exact KL (Analytical): {KL_exact:.6f} bits")
    
    gibbs_steps = [1000, 10000, 100000, 1000000]
    for steps in gibbs_steps:
        Q_gibbs = rbm_m4.run_gibbs_sampling(ALL_PATTERNS, n_steps=steps, burn_in=steps//10)
        kl_gibbs = calculate_kl_divergence(P_TRUE, Q_gibbs)
        print(f"  Gibbs sampling {steps:9d} steps: estimated KL = {kl_gibbs:.6f} bits")
        
    print("\nExperiment complete.")

