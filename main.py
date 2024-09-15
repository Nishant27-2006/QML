# Install necessary packages
!pip install torch pennylane scikit-learn matplotlib seaborn

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# QRNN Implementation
class QRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        n_qlayers=1,
        batch_first=True,
        backend="default.qubit"
    ):
        super(QRNN, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        self.wires = [f"wire_{i}" for i in range(self.n_qubits)]
        self.dev = qml.device(self.backend, wires=self.wires)

        def _layer_qrnn_block(W):
            def layer(W):
                for i in range(6):  # Apply rotations and CNOTs as per wire pairs
                    qml.RX(W[i, 0], wires=i)
                    qml.RZ(W[i, 1], wires=i)
                    qml.RX(W[i, 2], wires=i)

                for i in range(5):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(W[i + 1, 0], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])

        def _circuit_qrnn_block(inputs, weights):
            qml.AngleEmbedding(inputs, self.wires)
            for W in weights:
                _layer_qrnn_block(W)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.qlayer_circuit = qml.QNode(_circuit_qrnn_block, self.dev, interface="torch")

        weights_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        print(f"Weight Shapes: (n_qlayers, n_qubits, 3) = ({n_qlayers}, {n_qubits}, 3)")

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.VQC = qml.qnn.TorchLayer(self.qlayer_circuit, weights_shapes)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        q_input = self.clayer_in(combined)
        q_output = self.VQC(q_input)
        hidden = self.clayer_out(q_output)
        return hidden

# Preprocess stock data for QRNN
def preprocess_data(stock_data):
    numeric_columns = stock_data.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy="mean")
    stock_data[numeric_columns] = imputer.fit_transform(stock_data[numeric_columns])
    scaler = StandardScaler()
    stock_data[numeric_columns] = scaler.fit_transform(stock_data[numeric_columns])
    return stock_data

# Train the QRNN and plot loss curve
def train_qrnn(model, input_data, hidden_state, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_data, hidden_state)
        loss = criterion(output, torch.zeros_like(output))  # Dummy target
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# Plot training loss
def plot_loss_curve(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

# QRNN Model Output Plot
def plot_qrnn_output(output):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(output[0])), output[0].detach().numpy(), color='blue')
    plt.xlabel('Neuron Index')
    plt.ylabel('Output Value')
    plt.title('QRNN Hidden Layer Output')
    plt.show()

# Plot weight distribution after training
def plot_weight_distribution(model):
    weights = model.clayer_in.weight.detach().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(weights, bins=10, color='purple', alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution of QRNN')
    plt.show()

# Plot heatmap of QRNN output
def plot_activation_heatmap(output):
    plt.figure(figsize=(8, 6))
    sns.heatmap(output[0].detach().numpy().reshape(1, -1), annot=True, cmap='coolwarm', cbar=True)
    plt.xlabel('Neuron Index')
    plt.title('QRNN Hidden Layer Activation Heatmap')
    plt.show()

# Example input for QRNN (batch_size=1, input_size=3)
n_qubits = 5
n_qlayers = 2
qrnn = QRNN(input_size=3, hidden_size=2, n_qubits=n_qubits, n_qlayers=n_qlayers)
hidden_state = torch.zeros(1, qrnn.hidden_size)

input_data = torch.rand(1, 3)  # Random input for demonstration

# Train the model and plot training loss
losses = train_qrnn(qrnn, input_data, hidden_state, epochs=50)
plot_loss_curve(losses)

# Forward pass through QRNN and plot output
hidden_output = qrnn(input_data, hidden_state)
plot_qrnn_output(hidden_output)

# Plot weight distribution
plot_weight_distribution(qrnn)

# Plot heatmap of QRNN activations
plot_activation_heatmap(hidden_output)
