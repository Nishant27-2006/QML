# Standard library imports
import warnings

# Suppress specific warnings from PennyLane or other libraries if they are not critical
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

# Third-party library imports
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Added SimpleImputer for data preprocessing

# --- Quantum Recurrent Neural Network (QRNN) Implementation ---

class QRNN(nn.Module):
    """
    A Quantum Recurrent Neural Network (QRNN) module built with PyTorch and PennyLane.

    This QRNN integrates a Variational Quantum Circuit (VQC) within a classical neural network
    structure, enabling it to process sequential data with quantum enhancements.

    Args:
        input_size (int): The dimensionality of the input features.
        hidden_size (int): The dimensionality of the hidden state.
        n_qubits (int, optional): The number of qubits to use in the VQC. Defaults to 5.
        n_qlayers (int, optional): The number of quantum layers (blocks) in the VQC. Defaults to 1.
        batch_first (bool, optional): Not currently used, but kept for potential future
                                      compatibility with standard PyTorch RNN interfaces. Defaults to True.
        backend (str, optional): The PennyLane device backend to use for the QNode.
                                 Examples: "default.qubit", "lightning.qubit". Defaults to "default.qubit".
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        n_qlayers=1,
        batch_first=True,
        backend="default.qubit"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concatenated_size = self.input_size + self.hidden_size
        self.num_qubits = n_qubits
        self.num_quantum_layers = n_qlayers
        self.backend = backend

        # Initialize PennyLane device
        self.wires = [f"qwire_{i}" for i in range(self.num_qubits)]
        self.quantum_device = qml.device(self.backend, wires=self.wires)

        def _quantum_layer_block(weights):
            """
            Defines a single quantum layer block used within the VQC.
            This block consists of rotations and CNOT gates.

            Args:
                weights (torch.Tensor): A tensor of shape (num_qubits, 3) containing
                                        rotation parameters for each qubit.
            """
            for i in range(self.num_qubits):
                qml.RX(weights[i, 0], wires=i)
                qml.RZ(weights[i, 1], wires=i)
                qml.RX(weights[i, 2], wires=i)

            # Apply CNOT gates between adjacent qubits
            # The original code had a fixed range of 5 for CNOTs, assuming 6 qubits.
            # This is adjusted to be dynamic based on self.num_qubits for robustness.
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                # The original code applied RZ after CNOT, this might be part of a specific ansatz
                # or a typo. For general purpose, ensuring weights are indexed correctly is key.
                # Assuming weights are passed per layer, and each layer has enough weights for this.
                # If weights structure needs to be different, this part would require adjustment.
                # For now, keeping original logic for demonstration, but flagging for review.
                if i + 1 < self.num_qubits: # Ensure we don't go out of bounds for weights
                    qml.RZ(weights[i + 1, 0], wires=i + 1)
                qml.CNOT(wires=[i, i + 1]) # Another CNOT after RZ

        @qml.qnode(self.quantum_device, interface="torch")
        def _variational_quantum_circuit(inputs, vqc_weights):
            """
            The main Variational Quantum Circuit (VQC) that forms the core of the QRNN's
            quantum processing unit.

            Args:
                inputs (torch.Tensor): Classical input features encoded into the quantum state.
                vqc_weights (torch.Tensor): Trainable weights for the quantum layers.

            Returns:
                list[float]: Expected values of PauliZ operator on each qubit, serving as
                             the quantum circuit's output.
            """
            # Encode classical inputs into the quantum state
            qml.AngleEmbedding(inputs, wires=self.wires)

            # Apply multiple quantum layers
            for layer_weights in vqc_weights:
                _quantum_layer_block(layer_weights)

            # Measure expectation values
            return [qml.expval(qml.PauliZ(wire)) for wire in self.wires]

        self.variational_quantum_circuit = _variational_quantum_circuit

        # Define the shape of the weights for the VQC, required by TorchLayer
        vqc_weight_shapes = {"vqc_weights": (self.num_quantum_layers, self.num_qubits, 3)}
        print(f"VQC Weight Shapes: (num_quantum_layers, num_qubits, 3) = "
              f"({self.num_quantum_layers}, {self.num_qubits}, 3)")

        # Classical layers for input and output of the VQC
        self.classical_input_layer = nn.Linear(self.concatenated_size, self.num_qubits)
        self.quantum_layer = qml.qnn.TorchLayer(self.variational_quantum_circuit, vqc_weight_shapes)
        self.classical_output_layer = nn.Linear(self.num_qubits, self.hidden_size)

    def forward(self, x, hidden_state):
        """
        Defines the forward pass of the QRNN.

        Args:
            x (torch.Tensor): The current input features.
            hidden_state (torch.Tensor): The previous hidden state of the QRNN.

        Returns:
            torch.Tensor: The new hidden state after processing the input.
        """
        # Concatenate input and previous hidden state
        combined_input = torch.cat((x, hidden_state), 1)

        # Pass through classical input layer to prepare for quantum circuit
        quantum_circuit_input = self.classical_input_layer(combined_input)

        # Pass through the Variational Quantum Circuit
        quantum_circuit_output = self.quantum_layer(quantum_circuit_input)

        # Pass through classical output layer to produce the new hidden state
        new_hidden_state = self.classical_output_layer(quantum_circuit_output)
        return new_hidden_state

# --- Data Preprocessing Functions ---

def preprocess_stock_data(stock_data_df):
    """
    Preprocesses stock data by handling missing values and scaling numerical features.

    Args:
        stock_data_df (pd.DataFrame): The input DataFrame containing stock data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Identify numerical columns for imputation and scaling
    numeric_columns = stock_data_df.select_dtypes(include='number').columns

    # Impute missing numerical values with the mean
    imputer = SimpleImputer(strategy="mean")
    stock_data_df[numeric_columns] = imputer.fit_transform(stock_data_df[numeric_columns])

    # Scale numerical features
    scaler = StandardScaler()
    stock_data_df[numeric_columns] = scaler.fit_transform(stock_data_df[numeric_columns])
    return stock_data_df

# --- Training and Visualization Functions ---

def train_qrnn_model(model, input_data, initial_hidden_state, epochs=100, learning_rate=0.001):
    """
    Trains the QRNN model and records the loss over epochs.

    Args:
        model (QRNN): The QRNN model instance to train.
        input_data (torch.Tensor): The input data for training (e.g., a single batch).
        initial_hidden_state (torch.Tensor): The initial hidden state for the QRNN.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.

    Returns:
        list[float]: A list of loss values recorded at each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []

    print(f"\nStarting QRNN training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        # In a real scenario, `output` would be compared against a true target.
        # Here, it's a dummy target (zeros_like) for demonstration of training loop mechanics.
        current_output = model(input_data, initial_hidden_state)
        loss = criterion(current_output, torch.zeros_like(current_output))
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
    return training_losses

def plot_training_loss(losses):
    """
    Plots the training loss curve over epochs.

    Args:
        losses (list[float]): A list of loss values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='skyblue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('QRNN Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_qrnn_hidden_output(output_tensor):
    """
    Plots the values of the QRNN's hidden layer output as a bar chart.

    Args:
        output_tensor (torch.Tensor): The hidden layer output tensor from the QRNN.
                                     Expected to be a 2D tensor (batch_size, hidden_size).
    """
    # Detach from graph and convert to numpy for plotting
    output_values = output_tensor[0].detach().numpy() # Assuming batch_size=1 for this plot

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(output_values)), output_values, color='teal')
    plt.xlabel('Hidden Neuron Index')
    plt.ylabel('Activation Value')
    plt.title('QRNN Hidden Layer Output Activation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_model_weight_distribution(model_instance):
    """
    Plots the distribution of weights from the QRNN's classical input layer.

    Args:
        model_instance (QRNN): The trained QRNN model instance.
    """
    # Extract weights from the classical input layer (clayer_in)
    weights = model_instance.classical_input_layer.weight.detach().numpy().flatten()

    plt.figure(figsize=(8, 5))
    sns.histplot(weights, bins=20, kde=True, color='purple')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classical Input Layer Weights in QRNN')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_hidden_activation_heatmap(output_tensor):
    """
    Generates a heatmap visualization of the QRNN's hidden layer activations.

    Args:
        output_tensor (torch.Tensor): The hidden layer output tensor from the QRNN.
                                     Expected to be a 2D tensor (batch_size, hidden_size).
    """
    # Detach and reshape for heatmap visualization (assuming single sample for simplicity)
    activation_data = output_tensor.detach().numpy()

    plt.figure(figsize=(10, 3))
    sns.heatmap(activation_data, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Activation Value'})
    plt.xlabel('Hidden Neuron Index')
    plt.ylabel('Batch Sample' if activation_data.shape[0] > 1 else 'Single Sample')
    plt.title('QRNN Hidden Layer Activation Heatmap')
    plt.yticks(ticks=np.arange(activation_data.shape[0]) + 0.5, labels=[f'Sample {i+1}' for i in range(activation_data.shape[0])], rotation=0)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Initializing QRNN Model ---")

    # Define model parameters
    input_dim = 3  # Example: 3 features per input step
    hidden_dim = 2 # Example: 2 neurons in the hidden state
    num_qubits = 5
    num_quantum_layers = 2
    training_epochs = 75 # Reduced epochs for quicker demonstration

    # Create an instance of the QRNN model
    qrnn_model = QRNN(
        input_size=input_dim,
        hidden_size=hidden_dim,
        n_qubits=num_qubits,
        n_qlayers=num_quantum_layers
    )

    # Initialize a dummy input and hidden state for demonstration
    # In a real application, these would come from your dataset.
    # Batch size of 1 for this example.
    dummy_input_data = torch.rand(1, input_dim)
    initial_hidden_state = torch.zeros(1, qrnn_model.hidden_size)

    print(f"Dummy Input Data Shape: {dummy_input_data.shape}")
    print(f"Initial Hidden State Shape: {initial_hidden_state.shape}")

    # --- Train the QRNN Model ---
    print("\n--- Training QRNN ---")
    history_losses = train_qrnn_model(
        qrnn_model,
        dummy_input_data,
        initial_hidden_state,
        epochs=training_epochs
    )

    # --- Visualize Training Results ---
    print("\n--- Plotting Training Loss ---")
    plot_training_loss(history_losses)

    # --- Perform a Forward Pass and Visualize Output ---
    print("\n--- Performing Forward Pass and Plotting QRNN Output ---")
    # Get the final hidden output after training (or after a new forward pass)
    final_hidden_output = qrnn_model(dummy_input_data, initial_hidden_state)
    plot_qrnn_hidden_output(final_hidden_output)

    # --- Analyze Model Weights ---
    print("\n--- Plotting Model Weight Distribution ---")
    plot_model_weight_distribution(qrnn_model)

    # --- Visualize Hidden Layer Activations ---
    print("\n--- Plotting Hidden Layer Activation Heatmap ---")
    plot_hidden_activation_heatmap(final_hidden_output)

    print("\nQRNN demonstration complete. For real-world use, integrate with a proper dataset and task.")
