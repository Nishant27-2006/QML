#Quantum Recurrent Neural Network (QRNN) Implementation
This repository provides a basic implementation of a Quantum Recurrent Neural Network (QRNN) using Torch and PennyLane for quantum computing. The model is designed for processing time-series data, particularly in applications such as stock market prediction, where complex temporal dependencies exist.

Prerequisites
Before running the code, ensure that you have the necessary Python libraries installed:
pip install torch pennylane scikit-learn matplotlib seaborn
The code requires the following packages:

torch: for building and training the QRNN model.
pennylane: for constructing and simulating quantum circuits.
scikit-learn: for preprocessing the data (e.g., scaling and imputing missing values).
matplotlib: for plotting loss curves, weight distributions, and QRNN outputs.
seaborn: for visualizing activation heatmaps.
QRNN Overview
QRNN is a hybrid neural network combining classical and quantum computing elements. It is designed to process sequential data and capture dependencies over time by leveraging quantum circuits for feature extraction.

Components
Quantum Circuit Layer (VQC): A quantum variational layer embedded within the neural network, responsible for transforming classical input into quantum states and performing quantum operations.
Classical Layers: Fully connected layers to manage classical data inputs and outputs before and after the quantum operations.
QRNN Architecture
Input Size: The number of features in the input data.
Hidden Size: The size of the hidden layer for capturing temporal dependencies.
Number of Qubits: Specifies the number of qubits in the quantum circuit.
Quantum Layers: Defines the number of quantum layers applied.
Code Walkthrough
1. Model Implementation
The QRNN class is a PyTorch model, which includes the following components:

Input layer: A classical linear layer that preprocesses input data before feeding it into the quantum circuit.
VQC: A variational quantum circuit implemented using PennyLane's QNode and qml.qnn.TorchLayer.
Output layer: A classical linear layer that transforms the quantum outputs into the desired hidden state size.
2. Preprocessing Stock Data
The preprocess_data function uses scikit-learn to handle missing data and standardize the input values. It replaces missing values with the mean of the column and scales all numeric columns.
def preprocess_data(stock_data):
    numeric_columns = stock_data.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy="mean")
    stock_data[numeric_columns] = imputer.fit_transform(stock_data[numeric_columns])
    scaler = StandardScaler()
    stock_data[numeric_columns] = scaler.fit_transform(stock_data[numeric_columns])
    return stock_data
3. Training the QRNN
The train_qrnn function trains the QRNN model using Mean Squared Error (MSE) loss, and optimizes the parameters using Adam optimizer. After each epoch, the training loss is recorded for plotting.
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
4. Visualization
Several plotting functions are provided to visualize the results:

plot_loss_curve: Plots the training loss curve over epochs.
plot_qrnn_output: Displays a bar chart of the output values from the QRNN's hidden layer.
plot_weight_distribution: Plots the distribution of the weights in the input layer after training.
plot_activation_heatmap: Shows the activations of the hidden layer neurons as a heatmap.
def plot_loss_curve(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def plot_qrnn_output(output):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(output[0])), output[0].detach().numpy(), color='blue')
    plt.xlabel('Neuron Index')
    plt.ylabel('Output Value')
    plt.title('QRNN Hidden Layer Output')
    plt.show()

def plot_weight_distribution(model):
    weights = model.clayer_in.weight.detach().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(weights, bins=10, color='purple', alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution of QRNN')
    plt.show()

def plot_activation_heatmap(output):
    plt.figure(figsize=(8, 6))
    sns.heatmap(output[0].detach().numpy().reshape(1, -1), annot=True, cmap='coolwarm', cbar=True)
    plt.xlabel('Neuron Index')
    plt.title('QRNN Hidden Layer Activation Heatmap')
    plt.show()
Example Usage
The following demonstrates how to use the QRNN model:

Initialize the QRNN model:
qrnn = QRNN(input_size=3, hidden_size=2, n_qubits=5, n_qlayers=2)
hidden_state = torch.zeros(1, qrnn.hidden_size)
input_data = torch.rand(1, 3)  # Random input data

Train the QRNN model:
losses = train_qrnn(qrnn, input_data, hidden_state, epochs=50)
plot_loss_curve(losses)

Visualize QRNN outputs and weight distributions:
hidden_output = qrnn(input_data, hidden_state)
plot_qrnn_output(hidden_output)
plot_weight_distribution(qrnn)
plot_activation_heatmap(hidden_output)

Results
After training, the following visualizations will be produced:

Training Loss Curve: A plot showing how the model's loss decreases over epochs.
QRNN Output: A bar chart showing the values from the QRNN's hidden layer.
Weight Distribution: A histogram showing the distribution of weights after training.
Activation Heatmap: A heatmap visualizing the activations of QRNN neurons during a forward pass.

License
This project is licensed under the MIT License. Feel free to use and modify the code.



