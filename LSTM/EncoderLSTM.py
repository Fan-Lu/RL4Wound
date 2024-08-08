####################################################
# Description: LSTM autoencoder Implementation
# Version: V0.0.1
# Author: Sebastian Osorio @ UCSC
# Data: 2024-8-6
####################################################

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the ODE system
def ode_system(t, y, params):
    H, I, P, M = y
    k_h, k_i, k_p = params
    dHdt = -k_h * H
    dIdt = (k_h * H) - (k_i * I)
    dPdt = (k_i * I) - (k_p * P)
    dMdt = k_p * P
    return [dHdt, dIdt, dPdt, dMdt]

# Initial conditions
y0 = [0.9, 0.1, 0.0, 0.0]  # Initial values for y1, y2, y3, y4

# Time span
t_span = (0, 40)
t_eval = np.linspace(0, 40, 100)

# Parameters chosen by the best aunt ever
params = [0.2, 0.7, 0.3]

def generate_data(num_samples, t_span, t_eval, initial_conditions):
    data = []
    labels = []

    for _ in range(num_samples):
        params = np.random.rand(3)
        solution = solve_ivp(ode_system, t_span, initial_conditions, args=(params,), t_eval=t_eval)
        data.append(solution.y.T)
        labels.append(params)

    return np.array(data), np.array(labels)

num_samples =  100
t_span = (0,100)
t_eval = np.linspace(0, 100, 100)
initial_conditions = [0.9, 0.1, 0.0, 0.0]
data, labels = generate_data(num_samples, t_span, t_eval, initial_conditions)

print(f"data shape: {data.shape}")
print(f"labels shape: {labels.shape}") 

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        latent = self.hidden_to_latent(h[-1])
        return latent

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(LSTMDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, output_dim, batch_first=True)

    def forward(self, x, latent):
        print(f"Decoder input x shape: {x.shape}")
        print(f"Latent shape: {latent.shape}")
        h = self.latent_to_hidden(latent).unsqueeze(0).repeat( 1, x.size(0), 1)
        print(f"Hidden state shape: {h.shape}")
        c = torch.zeros_like(h)
        decoded, _ = self.lstm(x, (h,c))
        return decoded
    
class LSTMParameterPredictor(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(LSTMParameterPredictor, self).__init__()
        self.latent_to_params = nn.Linear(latent_dim, output_dim)

    def forward(self, latent):
        return self.latent_to_params(latent)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, param_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim)
        self.param_predictor = LSTMParameterPredictor(latent_dim, param_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(x, latent)
        params_pred = self.param_predictor(latent)
        return reconstructed, params_pred
    
input_dim = 4
hidden_dim = 64
latent_dim = 32
param_dim = 3
learning_rate = 0.001

model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, param_dim)
criterion_recon = nn.MSELoss()
criterion_params = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Testing

data_tensor = torch.tensor(data, dtype=torch.float32)
params_tensor = torch.tensor(labels, dtype=torch.float32)

print(f"data_tensor shape: {data_tensor.shape}")
print(f"params_tensor shape: {params_tensor.shape}")


batch_size = 32
dataset = TensorDataset(data_tensor, params_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


losses_recon = []
losses_params = []
total_losses = []

num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss_recon = 0
    epoch_loss_params = 0
    for batch_data, batch_params in dataloader:
        optimizer.zero_grad()
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch params shape: {batch_params.shape}")
        reconstructed, params_pred = model(batch_data)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Params predicted shape: {params_pred.shape}")

        loss_recon = criterion_recon(reconstructed, batch_data)
        loss_params = criterion_params(params_pred, batch_params)
        loss = loss_recon + loss_params
        
        loss.backward()
        optimizer.step()

        epoch_loss_recon += loss_recon.item()
        epoch_loss_params += loss_params.item()
    
    epoch_loss_recon /= len(dataloader)
    epoch_loss_params /= len(dataloader)
    total_loss = epoch_loss_recon + epoch_loss_params

    losses_recon.append(epoch_loss_recon)
    losses_params.append(epoch_loss_params)
    total_losses.append(total_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Recon Loss: {epoch_loss_recon:.4f}, Params Loss: {epoch_loss_params:.4f}, Total Loss: {total_loss:.4f}")

print("Training complete.")

plt.figure(figsize=(10,5))
plt.plot(losses_recon, label='Reconstruction Loss')
plt.plot(losses_params, label='Parameter Prediction Loss')
plt.plot(total_losses, label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.show()

model.eval()

with torch.no_grad():
    params_pred = []
    params_true = []
    for batch_data, batch_params in dataloader:
        _, batch_params_pred = model(batch_data)
        params_pred.append(batch_params_pred)
        params_true.append(batch_params)

params_pred = torch.cat(params_pred).cpu().numpy()
params_true = torch.cat(params_true).cpu().numpy()

mae = mean_absolute_error(params_true, params_pred)
rmse = mean_squared_error(params_true, params_pred, squared = False)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

for i in range(param_dim):
    plt.subplot(2, 2, i+1)
    plt.scatter(params_true[:, i], params_pred[:, i], alpha=0.5)
    plt.xlabel('True Parameter')
    plt.ylabel('Predicted Parameter')
    plt.title(f'Parameter {i+1}')

plt.tight_layout()
plt.show()