import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

# script to mimic inference that should happen on the FPGA

class SNN_Hardware_GoldenModel:
    def __init__(self):
        # Load Weights & Biases
        self.w1 = np.load("weights_fc1_int8.npy")   # shape [128, 64] expected
        self.w2 = np.load("weights_fc2_int8.npy")   # shape [64, 6]

        # Load Normalization Constants
        self.mean = np.load("scaler_mean.npy")
        self.scale = np.load("scaler_scale.npy")

        # 3. Define Hardware Constants
        self.beta_mult = 243  # 0.95 * 256 ~= 243
        self.beta_shift = 8   # Divide by 256

        # Thresholds derived from scaling:
        # Layer 1: Input(127 scale) * Weight(127 scale) = 16129 scale
        self.thresh1 = 16129  
        # Layer 2: Spike(1) * Weight(127 scale) = 127 scale
        self.thresh2 = 127

    def preprocess_hardware_input(self, raw_input):
        """
        Simulates the ARM Core processing before sending to FPGA.
        Returns: int8 array ready for FPGA
        """
        # Standard Scaler
        x_norm = (raw_input - self.mean) / (self.scale + 1e-10)
        
        # Clamp to [-3, 3] 
        x_clamped = np.clip(x_norm, -3.0, 3.0)
        
        # Scale to Int8 Range
        # map [-3, 3] -> [-1, 1] -> [-127, 127]
        x_scaled = (x_clamped / 3.0) * 127.0
        
        # Cast to Integer
        return np.round(x_scaled).astype(np.int8)

    def forward(self, x_raw, num_steps=30, update_weights=False):
        x_int8 = self.preprocess_hardware_input(x_raw)

        # neurons
        mem1 = np.zeros(self.w1.shape[1], dtype=np.int32)
        mem2 = np.zeros(self.w2.shape[1], dtype=np.int32)

        spike_counts = np.zeros(self.w2.shape[1], dtype=np.int32)

        # Pre-calculate Input Current for Layer 1
        # int8 * int8 = int32 result
        input_current_1 = np.dot(x_int8.astype(np.int32), self.w1.astype(np.int32))

        for t in range(num_steps):
            # LAYER 1 
            #  Decay
            mem1 = (mem1 * self.beta_mult) >> self.beta_shift
            
            # Integrate
            mem1 += input_current_1
            
            # Fire
            spikes1 = (mem1 > self.thresh1).astype(np.int32)

            # Reset
            mem1 -= (spikes1 * self.thresh1)

            # LAYER 2
            input_current_2 = np.dot(spikes1, self.w2.astype(np.int32))
            
            # Decay
            mem2 = (mem2 * self.beta_mult) >> self.beta_shift
            
            # Integrate
            mem2 += input_current_2
            
            # Fire
            spikes2 = (mem2 > self.thresh2).astype(np.int32)
            
            # Reset
            mem2 -= (spikes2 * self.thresh2)
            
            # Accumulate Output
            spike_counts += spikes2

        return spike_counts

def run_drift_analysis():
    model = SNN_Hardware_GoldenModel()

    # test data from training script (10 % of batch 1)
    X_data = np.load("X_test_raw.npy")
    y_data = np.load("y_test.npy")        

    # Inference
    correct = 0
    total = len(y_data)
        
    for j in range(total):
        spike_counts = model.forward(X_data[j], num_steps=30, update_weights=True)
        prediction = np.argmax(spike_counts)
        if prediction == y_data[j]:
            correct += 1
        
    # Record Stats
    acc = 100 * correct / total
    
    print(f"Batch 1 | {total:<10} | {acc:.2f}%")


if __name__ == "__main__":
    run_drift_analysis()