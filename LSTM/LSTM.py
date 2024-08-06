####################################################
# Description: LSTM Implementation
# Version: V0.0.1
# Author: Sebastian Osorio @ UCSC
# Data: 2024-7-16
####################################################

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

print("This is our data: ", data.shape)

print("This is our labels: ", labels.shape)

# Reshaping our data for LSTM model
data = data.reshape(num_samples, t_eval.size, 4)

# Creating our tensors for the model
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Dataset creation
dataset = TensorDataset(data_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

hidden_size = 64

# Now we just create LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size,  hidden_size, output_size, num_layers = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
input_size = 4
hidden_size = 128
output_size = 3
model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    train_losses.append(running_train_loss / len(train_loader))
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()
    
    val_losses.append(running_val_loss / len(val_loader))
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses over Epochs')
plt.legend()
plt.show()

num_test_samples = 200
test_data, test_labels = generate_data(num_test_samples, t_span, t_eval, initial_conditions)

test_data_tensor = torch.tensor(test_data, dtype = torch.float32).to(device)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predicted_params = model(test_data_tensor)

predicted_params_np = predicted_params.cpu().numpy()
test_labels_np = test_labels_tensor.cpu().numpy()

for i in range(5):
    print(f"Sample {i + 1}:")
    print(f"True Parameters: {test_labels_np[i]}")
    print(f"Predicted Parameters: {predicted_params_np[i]}")
    print()

mse = mean_squared_error(test_labels_tensor.cpu().numpy(), predicted_params.cpu().numpy())
print(f'Mean Squared Error on Test Data: {mse}')