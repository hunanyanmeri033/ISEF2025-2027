#
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print("-----------------------------------------------------------------------------------")



def variational_layer(n_qubits, weights, layer):
    """
    Applies a general qubit rotation to each qubit
    """
    for q in range(n_qubits):
        phi, theta, omega = weights[layer, 0, q]
        qml.Rot(phi, theta, omega, wires=q)

def qnn_angle_encoding(n_qubits=4, n_layers=4):
    dev = qml.device("default.mixed", wires=n_qubits)
    @qml.qnode(dev, interface='torch')
    def qnode(inputs, weights):  # HAVE to have a parameter with the name inputs for your qnode. 
        #print("before lopp")
        for l in range(n_layers):
            #print("inside loop")
            qml.templates.AngleEmbedding(inputs[:, l*n_qubits:(l+1)*n_qubits], wires=range(n_qubits), rotation="Y")
            #print("after angle")
            qml.templates.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            #print("after entangling")
            #if l == n_layers - 1:
            #    print("in if statement")
            #    variational_layer(n_qubits, weights, l)  # final variational layer

        """for X in inputs:
            print("before splitting into chunks")
            x_chunks = [] 
            n_layers = weights.shape[0]
            print("input shape")
            print(X.shape)
            #print('X', x)
            for i in range(0, len(X), 5): # partitioning the latent vector
                x_chunks.append(X[i:i+6])
            print("after chunks")
            #print("hello-----------------------------------------------")
            qml.AngleEmbedding(features=x_chunks[n_layers % len(x_chunks)], wires=range(n_qubits), rotation='Y')# encoding 
            qml.templates.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            print("after loop")"""
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        #return qml.expval(H)
    
    single_layer_shape = qml.templates.StronglyEntanglingLayers.shape(
        n_layers=n_layers, n_wires=n_qubits)

    # Transform qnode to torch layer
    # batched_qnode = broadcast_expand(qnode)
    qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
    with torch.no_grad():
        qlayer.weights.normal_(mean=0, std=0.01 * np.pi)
    return qlayer

class Scale(nn.Module):
    """
    Scales the input by a constant factor
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class Hybrid2DConvNet(nn.Module):
    def __init__(self, latent_dim, pre_layers="3conv", latent_activation="tanh", encoding_type="amplitude", entanglement=True, observable="global", freeze_qnn=False, noise=None):
        super().__init__()



        self.pre_layers = pre_layers
        self.observable = observable


        self.norm1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.norm2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)


        self.norm3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        if self.pre_layers == "3conv":
            self.pre = nn.Sequential(
                nn.Linear(256, latent_dim), #In-features, out-features
                nn.BatchNorm1d(latent_dim))
        elif self.pre_layers == "2conv":
            self.pre = nn.Sequential(
                nn.Linear(512, latent_dim),
                nn.BatchNorm1d(latent_dim))
        elif self.pre_layers == "1conv":
            self.pre = nn.Sequential(
                nn.Linear(256, latent_dim),
                nn.BatchNorm1d(latent_dim))
        elif self.pre_layers == "0conv":
            self.pre = nn.Sequential(
                nn.Linear(256, latent_dim),
                nn.BatchNorm1d(latent_dim))

        if encoding_type == "angle":
            if latent_activation == "tanh":
                self.pre.add_module("tanh", nn.Tanh())
                self.pre.add_module("scale", Scale(np.pi))

            self.qnn = qnn_angle_encoding(n_qubits=6, n_layers=4)
        
        if freeze_qnn:
            for param in self.qnn.parameters():
                param.requires_grad = False
        
        self.out = nn.Linear(6, 1)

    def forward(self, x):
        if self.pre_layers == "3conv":
            x = self.pool1(self.relu1(self.norm1(x)))
            x = self.pool2(self.relu2(self.norm2(x)))
            x = self.pool3(self.relu3(self.norm3(x)))

        elif self.pre_layers == "2conv":
            x = self.pool1(self.relu1(self.norm1(x)))
            x = self.pool2(self.relu2(self.norm2(x)))

        elif self.pre_layers == "1conv":
            x = self.pool1(self.relu1(self.norm1(x)))

        x = x.view(x.size(0), -1)
        x = self.pre(x)
        print("hello---------------------------------------------pree")

        x = self.qnn(x)

    
        x = self.out(x)
        return x

X_train = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/X_trainq_tf.npy"))#data/processed/X_test_tf.npy
y_train = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/y_trainq_tf.npy"))

X_test = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/X_testq_tf.npy"))
y_test = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/y_testq_tf.npy"))

X_valid = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/X_valq_tf.npy"))
y_valid = torch.tensor(np.load("/workspaces/ISEF2025-2027/data/processed/y_valq_tf.npy"))
"""y_train = 2 * y_train - 1
y_test  = 2 * y_test  - 1
y_valid = 2 * y_valid - 1"""
n_qubits = 6 # change to pca values
dev = qml.device("default.qubit", wires=n_qubits)
H = qml.Hamiltonian(
    coeffs=[1/n_qubits] * n_qubits,
    observables=[qml.PauliZ(i) for i in range(n_qubits)]
)
dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
dataset_validation = torch.utils.data.TensorDataset(X_valid, y_valid)
dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

batch_size = 10 # -------------------------------------------------------------------------batch_size
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
num_layers = 4
params = [torch.rand(num_layers, n_qubits, 3)]
model = Hybrid2DConvNet(6, "2conv", "tanh", "angle", False, "global", False, None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
result_list = []
num_epochs = 80
total_loss=0
total_accuracy=0
total_validation_loss=0
total_validation_accuracy=0
preds = 0
for epoch in range(num_epochs):
    print("epoch"+str(epoch))
    total_loss = 0.0
    total_accuracy = 0.0
    total_validation_loss = 0.0
    total_validation_accuracy = 0.0
    for batch_inputs, batch_labels in dataloader_train:
        train_preds = model(batch_inputs)
        #print(train_preds)
        batch_labels = batch_labels.float().view(-1,1)
        loss = criterion(train_preds, batch_labels)
        train_loss = criterion(train_preds, batch_labels)
        #print(train_loss)
        total_loss += train_loss.item() * batch_inputs.shape[0]
        probs = torch.sigmoid(train_preds)
        preds = (probs > 0.5).float()
        print(f"train_loss{train_loss}")
        print(f"total_loss{total_loss}")
        total_accuracy += (
            preds == batch_labels
        ).float().mean().item()
        #print(total_accuracy)
        #total_accuracy += (train_preds.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item() / batch_labels.size()[0]
        #print(total_accuracy)
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
    avg_train_loss = total_loss / len(dataset_train)
    avg_train_accuracy = total_accuracy / len(dataloader_train)

    with torch.no_grad():
            for batch_inputs, batch_labels in dataloader_validation:
                val_predictions = model(batch_inputs)
                batch_labels = batch_labels.float().view(-1,1) # convert to 2d
                val_loss = criterion(val_predictions, batch_labels)
                total_validation_loss += val_loss.item() * batch_inputs.shape[0]
                #print(f"batch_label size: {batch_labels.size()[0]}")
                probs = torch.sigmoid(val_predictions)
                preds = (probs > 0.5).float()
                total_validation_accuracy += (
                    preds == batch_labels
                ).float().mean().item()
                #total_validation_accuracy += (val_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item() / batch_labels.size()[0]
    
    avg_val_loss = total_validation_loss / len(dataset_validation)
    avg_val_accuracy = total_validation_accuracy / len(dataloader_validation)


    # Print the progress
    print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}, Train accuracy = {avg_train_accuracy:.4f},  Validation accuracy = {avg_val_accuracy:.4f}")

# Add values to list
    result_list.append((epoch+1, avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy, model.state_dict()))

lowest_val = [0, 100, 0, 0]
for element in result_list:
    if element[2] < lowest_val[1]:
        lowest_val[:] = [element[5], element[2], element[3], element[4]]

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
print(f"parameters{lowest_val[0]}")
model.load_state_dict(lowest_val[0])
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X_test, y_test in test_loader: # Assuming you have a test DataLoader
        
        # Forward pass: get predictions
        outputs = model(X_test)
        
        # Get the predicted class (for classification tasks)
        _, predicted = torch.max(outputs.data, 1)
        
        # Accumulate metrics
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

# 4. Calculate Final Metric
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

print(lowest_val)

print("------------DOne______________________________________")









