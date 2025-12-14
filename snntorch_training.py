import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import snntorch as snn
import snntorch.functional as SF

import numpy as np
import os

def load_batch(filename):
    print(f"Loading {filename}...")

    # load_svmlight_file automatically handles the "1:val 2:val" format
    X_sparse, y = load_svmlight_file(filename, n_features=128)
    
    # convert to numpy array
    X = X_sparse.toarray()

    # adjust labels to be 0-indexed
    y = (y-1).astype(int)

    return X, y

def get_data_loader(batch_file,test_ratio=0.1, batch_size=64):

    # load in single file 
    if not os.path.exists(batch_file):
        print(f"Warning: {batch_file} not found. Skipping.")
        
    X, y = load_batch(batch_file)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=y,
            shuffle=True,
            random_state=42
    )

    # Save 10% validation before scaling to verify the hardware's scaling logic later
    np.save("X_test_raw.npy", X_test)
    np.save("y_test.npy", y_test)
    print(f"Saved {len(y_test)} raw test samples to X_test_raw.npy / y_test.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Save normalization constants
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    print("Normalization parameters saved.")

    X_test = scaler.transform(X_test)
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=False
    )


    return train_loader, test_loader

class GasSensorSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()
        
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps=30):
        
        x = torch.clamp(x, min=-3.0, max=3.0)

        x_norm = x / 3.0

        # 3. "Fake" Quantize weights during forward pass
        #    This forces the network to learn weights that work well when clamped.
        #    We use tanh to squash weights between -1 and 1.
        w1_fake = torch.tanh(self.fc1.weight)
        w2_fake = torch.tanh(self.fc2.weight)

        # Initialize hidden states 
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer spikes
        spk2_rec = []
        
        # Time Loop
        for step in range(num_steps):
            # Layer 1
            # cur1 = self.fc1(x) # Input x is constant (Direct Encoding)
            cur1 = nn.functional.linear(x_norm, w1_fake)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            # cur2 = self.fc2(spk1)
            cur2 = nn.functional.linear(spk1, w2_fake)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec, dim=0)

if __name__ == "__main__":
    BATCH_FILE = './Dataset/batch1.dat'

    #  Load Data
    train_loader, test_loader = get_data_loader(BATCH_FILE)
    print("Data loaded successfully.")
    
    # Instantiate Model
    model = GasSensorSNN(num_inputs=128, num_hidden=64, num_outputs=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = SF.ce_rate_loss()

    print(f"Starting training on cpu...")

    best_acc = 0.0

    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            # Forward pass
            spk_rec = model(data)
            # Loss Calculation
            loss_val: torch.Tensor = criterion(spk_rec, targets)
            
            # Gradient Step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # Accuracy Tracking
            epoch_loss += loss_val.item()
            # Sum spikes over time to get prediction index
            predicted = spk_rec.sum(dim=0).argmax(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                spk_rec = model(data)
                preds = spk_rec.sum(dim=0).argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Test Acc: {acc:.2f}%")

        # --- SAVE ONLY IF IT IMPROVES ---
        if acc > best_acc:
            best_acc = acc
            print(f"--> New Best Model! Saving... ({best_acc:.2f}%)")
            
            # Save the integer-ready weights immediately
            w1 = model.fc1.weight.detach().cpu()
            w2 = model.fc2.weight.detach().cpu()
            
            # Apply your quantization/clamping logic here before saving
            w1_int8 = (torch.tanh(w1) * 127).round().numpy().astype(np.int8).T
            w2_int8 = (torch.tanh(w2) * 127).round().numpy().astype(np.int8).T
            
            # np.save("weights_fc1_int8.npy", w1_int8)
            # np.save("weights_fc2_int8.npy", w2_int8)
            # torch.save(model.state_dict(), "best_model.pth")
            
    print(f"Training Complete. Best Accuracy achieved: {best_acc:.2f}%")