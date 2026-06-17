import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import random 
import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import math
import os
import csv
import sys

X_train = np.load("../data/processed/X_trainq.npy")
y_train = np.load("../data/processed/y_trainq_tf.npy")

X_test = np.load("../data/processed/X_testq.npy")
y_test = np.load("../data/processed/y_testq_tf.npy")

X_valid = np.load("../data/processed/X_valq.npy")
y_valid = np.load("../data/processed/y_valq_tf.npy")
y_train = 2 * y_train - 1
y_test  = 2 * y_test  - 1
y_valid = 2 * y_valid - 1
n_qubits = 6 # change to pca values
dev = qml.device("default.qubit", wires=n_qubits)

H = qml.Hamiltonian(
    coeffs=[1/n_qubits] * n_qubits,
    observables=[qml.PauliZ(i) for i in range(n_qubits)]
)


def Hadamard(nqubits):
    return [qml.Hadamard(wires=idx) for idx in range(nqubits)]

def loss(params, X, Y, lam=5e-4):
    preds = np.array([variational_circuit(params, x) for x in X])
    reg = lam * np.sum(params**2)
    #preds = np.tanh(5*preds)
    return np.mean(np.log(1 + np.exp(-Y * preds)))

def Entangle(params): 
    return [[*CNot(0, len(p)), *CNot(1, len(p)), *Rotation(p)] for p in params]

def Rotation(p):
    return [qml.RY(element, wires=idx) for idx, element in enumerate(p)]

def CNot(start, nqubits):
    return [qml.CNOT(wires=[i, i + 1]) for i in range(start, nqubits - 1, 2)] 

def Measure(wires):
    return [qml.expval(qml.PauliZ(position)) for position in wires]  

@qml.qnode(dev, interface="torch", diff_method='adjoint')
def variational_circuit(params, x, out):
    print(params)
    params =  2.0 * torch.arctan(2 * params)# weight remapping
    print("type inside vqc")
    print(type(params))
    width = params.shape[1]    
    #assert input.shape[0] == width, f"Expected input of len {width}"
    #input = input * np.pi - np.pi / 2.0   # Rescale [0, 1] to [-pi/2, pi/2]
    Hadamard(width)               # Start from state |+> , unbiased w.r.t. |0> and |1>
    print("before angleembedding")
    #qml.AngleEmbedding(features=[x], wires=range(n_qubits), rotation= 'Y')  # Embed features in the quantum node 
    Rotation(x)
    Entangle(params)# Sequence of trainable variational layers
    return qml.expval(H) 

def accuracy(params, X, y):
    preds = [variational_circuit(params, data) for data in X]
    predictions = np.sign(np.stack(predictions))
    return np.mean(predictions == Y)

def test_model(model, dataset_test, batch_size):
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    model.eval()
    results_list = []
    test_loss = 0.0
    total_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader_test:
            test_predictions = model(batch_inputs)
            y_probs = torch.softmax(test_predictions, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)
            y_trues = torch.argmax(batch_labels, dim=1)

            for i in range(len(batch_inputs)):
                sample_result = (y_trues[i].item(), y_preds[i].item(), y_probs[i].tolist())
                results_list.append(sample_result)

            # Compute the test loss
            loss = criterion(test_predictions, y_trues)
            test_loss += loss.item() * batch_inputs.shape[0]

            # Compute the test accuracy
            total_accuracy += (y_preds == y_trues).sum().item()

    # Calculate average test loss and test accuracy
    avg_test_loss = test_loss / len(dataset_test)
    avg_test_accuracy = total_accuracy / len(dataset_test)


    return results_list, avg_test_loss, avg_test_accuracy


class Circuit(nn.Module):
    def __init__(self, width, depth, out, amplitude_embedding):
        super().__init__()
        self.out = out
        self.params = torch.nn.Parameter(torch.randn(depth, width))
        self.amplitude_embedding = amplitude_embedding

    def forward(self, input):
        if len(input.shape) > 1: return torch.cat([self(i).float().unsqueeze(0) for i in input])
        if self.amplitude_embedding:
            return variational_circuit_amplitude_embedding(input, self.params, self.out).float()
        else:
            #print(variational_circuit(self.params, input, self.out))
            return variational_circuit(self.params, input, self.out).float()

def train_and_validate_model(model, dataset_train, dataset_validation, batch_size, num_epochs, learning_rate):    
    # Define the dataloader for efficient batch training
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    result_list = []
    
    best_val_acc=0

    for epoch in range(num_epochs):
        # Initialize the total loss and accuracy for this epoch
        total_loss = 0.0
        total_accuracy = 0.0

        start_time = time.time()

        # Train the model
        model.train()

        # Loop over the batches
        for batch_inputs, batch_labels in dataloader_train:
            # Reset the gradients
            optimizer.zero_grad()
            # Compute the predictions
            train_predictions = model(batch_inputs)
            # Compute the loss
            loss = criterion(train_predictions, batch_labels.argmax(dim=1))
            # Accumulate the loss and accuracy
            total_loss += loss.item() * batch_inputs.shape[0]
            total_accuracy += (train_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
            # Compute the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
        
        # Compute the average loss and accuracy for this epoch
        avg_train_loss = total_loss / len(dataset_train)
        avg_train_accuracy = total_accuracy / len(dataset_train)

        # Evaluate the model on the validation set
        model.eval()
        total_validation_loss = 0.0
        total_validation_accuracy = 0.0

        with torch.no_grad():
            for batch_inputs, batch_labels in dataloader_validation:
                val_predictions = model(batch_inputs)
                val_loss = criterion(val_predictions, batch_labels.argmax(dim=1))
                total_validation_loss += val_loss.item() * batch_inputs.shape[0]
                total_validation_accuracy += (val_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
        
        # Compute the average validation loss and accuracy for this epoch
        avg_val_loss = total_validation_loss / len(dataset_validation)
        avg_val_accuracy = total_validation_accuracy / len(dataset_validation)

        end_time = time.time()
        epoch_time = end_time - start_time
        # Print the progress
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}, Train accuracy = {avg_train_accuracy:.4f},  Validation accuracy = {avg_val_accuracy:.4f}, Epoch Time: {epoch_time:.2f} seconds")
        if avg_val_accuracy>best_val_acc:
            best_val_acc = avg_val_accuracy
            best_state = copy.deepcopy(model.state_dict())
        # Add values to list
        result_list.append((epoch+1, avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy, epoch_time))
    model.load_state_dict(best_dict) 
    return result_list



num_layers = 4
num_epochs = 300
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

epoch_list = []
loss_list = []
train_acc_list = []
test_acc_list = []
val_params = []
valid_acc_list = []
print(type(X_train))
print(type(y_train))
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_valid = torch.tensor(X_valid)
y_valid = torch.tensor(y_valid)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)
print(type(X_train))
print(type(y_train))
X_train = X_train.to(torch.float32)
X_valid = X_valid.to(torch.float32)
X_test = X_test.to(torch.float32)

y_train = y_train.to(torch.float32)
y_valid = y_valid.to(torch.float32)
y_test = y_test.to(torch.float32)

train_d = torch.utils.data.TensorDataset(X_train, y_train)
valid_d = torch.utils.data.TensorDataset(X_valid, y_valid)
test_d = torch.utils.data.TensorDataset(X_test, y_test)
#batch_sizetr = len(train_d)
#batch_sizeva = len(valid_d)
#batch_sizete = len(test_d)
circuit1 = Circuit(6, num_layers, 2,  amplitude_embedding=False)

#circuit2 = Circuit(6, num_layers, 2,  amplitude_embedding=True)

#angle embedding
VQC1 = train_and_validate_model(circuit1, train_d, valid_d, 100, 300, 0.03)

#tets model
results1 = test_model(VQC1, test_d, 100)
print(results1[2])

#amplitude embedding
VQC2 = train_and_validate_model(circuit2, train_d, valid_d, 100, 300, 0.03)

#test model
results2 = test_model(VQC2, test_d, 100)
print(results2[2])
