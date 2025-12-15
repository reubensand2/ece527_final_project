import numpy as np

# Creates the data file for the c-sim testbench

# test data from split
X_test_raw = np.load("X_test_raw.npy") 
y_test = np.load("y_test.npy")

# weights
w1 = np.load("weights_fc1_int8.npy").astype(np.int8) 
w2 = np.load("weights_fc2_int8.npy").astype(np.int8)

# scale
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

# LOAD & QUANTIZE INPUT
def get_quantized_samples(raw_data):
    # 1. Standard Scaler
    x_norm = (raw_data - mean) / (scale + 1e-10)
    # 2. Clamp [-3, 3]
    x_clamped = np.clip(x_norm, -3.0, 3.0)
    # 3. Scale & Cast
    x_int8 = np.round((x_clamped / 3.0) * 127.0).astype(np.int8)
    
    return x_int8

X_test_int8 = get_quantized_samples(X_test_raw)
num_samples = len(y_test)

# GENERATE C HEADER
with open("tb_data.h", "w") as f:
    f.write("#ifndef TB_DATA_H\n#define TB_DATA_H\n\n")
    f.write("#include \"ap_int.h\"\n\n")

    # FLATTENED WEIGHTS
    weights_flat = np.concatenate([w1.flatten(), w2.flatten()])
    f.write(f"// Total Weights: {len(weights_flat)}\n")
    f.write("const ap_int<8> TB_WEIGHTS[] = {\n    ")
    for i, w in enumerate(weights_flat):
        f.write(f"{w}, ")
        if (i+1) % 16 == 0: f.write("\n    ")
    f.write("\n};\n\n")

    # INPUT SAMPLES
    f.write(f"// {num_samples} Test Samples from Batch 1\n")
    f.write(f"const ap_int<8> TB_INPUTS[{num_samples}][128] = {{\n")
    for row in X_test_int8:
        f.write("    {" + ", ".join(map(str, row)) + "},\n")
    f.write("};\n\n")

    # EXPECTED LABELS
    f.write("const int TB_LABELS[] = {" + ", ".join(map(str, y_test)) + "};\n\n")
    
    f.write("#endif")

print(f"Generated tb_data.h with {num_samples} samples.")