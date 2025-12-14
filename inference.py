import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import os

import numpy as np

class SNN_Hardware_GoldenModel:
    def __init__(self):
        # self.beta = beta
        # self.threshold = threshold

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
        # A. Standard Scaler (Float math on ARM)
        x_norm = (raw_input - self.mean) / (self.scale + 1e-10)
        
        # B. Clamp to [-3, 3] (Float math on ARM)
        x_clamped = np.clip(x_norm, -3.0, 3.0)
        
        # C. Scale to Int8 Range (Float math on ARM)
        # map [-3, 3] -> [-1, 1] -> [-127, 127]
        x_scaled = (x_clamped / 3.0) * 127.0
        
        # D. Cast to Integer (This is what goes over AXI stream)
        return np.round(x_scaled).astype(np.int8)

    def forward(self, x_raw, num_steps=30, update_weights=False):
        x_int8 = self.preprocess_hardware_input(x_raw)

        # neurons
        mem1 = np.zeros(self.w1.shape[1], dtype=np.int32)
        mem2 = np.zeros(self.w2.shape[1], dtype=np.int32)

        spike_counts = np.zeros(self.w2.shape[1], dtype=np.int32)

        # Pre-calculate Input Current for Layer 1 (Direct Encoding)
        # int8 * int8 = int32 result
        input_current_1 = np.dot(x_int8.astype(np.int32), self.w1.astype(np.int32))

        for t in range(num_steps):
            #  Decay
            mem1 = (mem1 * self.beta_mult) >> self.beta_shift
            
            # 2. Integrate
            mem1 += input_current_1
            
            # 3. Fire
            spikes1 = (mem1 > self.thresh1).astype(np.int32)

            # 4. Reset (Soft Reset)
            mem1 -= (spikes1 * self.thresh1)

            # --- LAYER 2 ---
            # Incoming Current: 1 (spike) * Weight
            # Effectively summing weights where spikes1 is 1
            input_current_2 = np.dot(spikes1, self.w2.astype(np.int32))
            
            # 1. Decay
            mem2 = (mem2 * self.beta_mult) >> self.beta_shift
            
            # 2. Integrate
            mem2 += input_current_2
            
            # 3. Fire
            spikes2 = (mem2 > self.thresh2).astype(np.int32)
            
            # 4. Reset
            mem2 -= (spikes2 * self.thresh2)
            
            # Accumulate Output
            spike_counts += spikes2

        return spike_counts

def run_drift_analysis():
    # List of all batch files
    # Adjust path './Dataset/' to match your folder structure exactly
    # batch_files = [f'./Dataset/batch{i}.dat' for i in range(1, 11)]
    # BATCH_FILE = './Dataset/batch1.dat'

    model = SNN_Hardware_GoldenModel()

            
    # 1. Load Data
    # X_sparse, y = load_svmlight_file(BATCH_FILE, n_features=128)
    # X_data = X_sparse.toarray()
    # y_data = y - 1 

    X_data = np.load("X_test_raw.npy")
    y_data = np.load("y_test.npy")        

    # 2. Run Inference
    correct = 0
    total = len(y_data)
        
    for j in range(total):
        spike_counts = model.forward(X_data[j], num_steps=30, update_weights=True)
        prediction = np.argmax(spike_counts)
        if prediction == y_data[j]:
            correct += 1
        
    # 3. Record Stats
    acc = 100 * correct / total
    
    print(f"Batch 1 | {total:<10} | {acc:.2f}%")


if __name__ == "__main__":
    run_drift_analysis()