import numpy as np
from sklearn.datasets import load_svmlight_file

NUM_INPUTS = 128
NUM_HIDDEN = 64
NUM_OUTPUTS = 6     # 6 gasses
TIME_STEPS = 50     # simulation window per sample
TAU = 0.9           # membrane decay factor
V_THRESH = 50       # threshold to spike (tunable)

def load_and_encode_batch(filename):
    """
    Loads LibSVM and converts 128 floats into 128 spike times (latency encoding)
    
    :param filename: file containing data
    """
    X_sparse, y = load_svmlight_file(filename, n_features=128)
    X = X_sparse.toarray()

    # normalize features to 0-1 range for easier encoding
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-6)

    # LATENCY ENCODING STRATEGY
    # Higher Value = Earlier Spike (Low Latency)
    # Lower Value = Later Spike (High Latency)
    # Value 0.0 -> Spike at t=50 (or never)
    # Value 1.0 -> Spike at t=0

    # Invert: 1.0 - X_norm gives low values for high inputs
    # Scale to Time Steps: (1.0 - X_norm) * TIME_STEPS
    latency_codes = np.floor((1.0 - X_norm) * TIME_STEPS).astype(int)
    
    # Adjust labels to be 0-5 instead of 1-6 for array indexing
    y = y - 1 
    
    return latency_codes, y

class BareMetalSNN:
    def __init__(self):
        self.W1 = np.random.uniform(0, 1.0, (NUM_INPUTS, NUM_HIDDEN))
        self.W2 = np.random.uniform(0, 1.0, (NUM_HIDDEN, NUM_OUTPUTS))

    def forward_pass(self, spike_times):

        # membrane potentials (registers in hardware)
        v_hidden = np.zeros(NUM_HIDDEN)
        v_output = np.zeros(NUM_OUTPUTS)

        # Output Spike Counter (Rate Decoding)
        out_spikes = np.zeros(NUM_OUTPUTS)

        for t in range(TIME_STEPS):
            # 1. INPUT LAYER
            # Which input neurons fire at this exact tick 't'?
            # This mimics the "Spike Generator" block on FPGA
            active_inputs = (spike_times == t) 
            
            # 2. HIDDEN LAYER
            # Accumulate incoming current from active inputs
            # Logic: If input[i] spikes, add W1[i] to hidden neurons
            input_current = np.sum(self.W1[active_inputs], axis=0)
            
            # Update Membrane Potential (Leaky Integrate)
            v_hidden = (v_hidden * TAU) + input_current
            
            # Fire & Reset
            fired_hidden = v_hidden >= V_THRESH
            v_hidden[fired_hidden] = 0.0 # Hard Reset
            
            # 3. OUTPUT LAYER
            # Accumulate current from firing hidden neurons
            hidden_current = np.sum(self.W2[fired_hidden], axis=0)
            
            # Update Membrane Potential
            v_output = (v_output * TAU) + hidden_current
            
            # Fire & Reset
            fired_output = v_output >= V_THRESH
            v_output[fired_output] = 0.0
            
            # Count Spikes (for classification)
            out_spikes[fired_output] += 1
            
        return np.argmax(out_spikes)
    
X_b1, y_b1 = load_and_encode_batch("./Dataset/batch1.dat")