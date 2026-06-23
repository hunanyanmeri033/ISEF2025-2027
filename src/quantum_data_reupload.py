#quantum data collection code"
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

params = np.load("/workspaces/ISEF2025-2027/models/quantum_params.npy", allow_pickle=True)
print("Quantum model parameters loaded.")

#metadata = np.load("quantum_model_metadata.npy", allow_pickle=True).item()
#print(metadata)

def accuracy(params, X, Y):
    predictions = [variational_circuit(params, x) for x in X]
    predictions = np.sign(np.stack(predictions))
    return np.mean(predictions==Y)
X_test = np.load("/workspaces/ISEF2025-2027/data/processed/X_testq_tf.npy")
y_test = np.load("/workspaces/ISEF2025-2027/data/processed/y_testq_tf.npy")
y_test  = 2 * y_test  - 1
n_qubits = 6

dev = qml.device("default.qubit", wires=n_qubits)

H = qml.Hamiltonian(
        coeffs = [1/n_qubits] * n_qubits,
        observables=[qml.PauliZ(i) for i in range(n_qubits)]
)


@qml.qnode(dev)
def feature_encoding(finger):
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='Y')
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]

@qml.qnode(dev, interface='autograd')
def variational_circuit(params, x):
    x_chunks = []
    for i in range(0, len(x), 5):
        x_chunks.append(x[i:i+6])
    # params shape: (num_layers, n_qubits, 3)
    # Each qubit in each layer has 3 parameters for RX, RY, and RZ rotations
    for i_layer, layer_params in enumerate(params):
        #print("hello-----------------------------------------------")
        qml.AngleEmbedding(features=x_chunks[i_layer % len(x_chunks)], wires=range(n_qubits), rotation='Y')# encoding 
        #print("after angle embedding --------------------------------------------------------------------")
        #for i, wire_params in enumerate(layer_params): # rotation
        #    qml.RY(wire_params, wires=i)
        #print(layer_params.shape)
        qml.StronglyEntanglingLayers(weights=params, wires=range(6))

    # Measure expectation value of Pauli-Z on the first qubit
    return qml.expval(H)

def predict_raw(params, X):
    return np.array([variational_circuit(params, x) for x in X])

def predict_label(raw_preds):
    return np.where(raw_preds >= 0, 1, -1)

raw_preds = predict_raw(params, X_test)
y_pred = predict_label(raw_preds)
y_test_auc = (y_test + 1) // 2   # {-1,1} → {0,1}


#y_pred = accuracy(params, X_test, y_test)
print(y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test_auc, raw_preds)
accuracy = accuracy(params, X_test, y_test)
print(f1)
print(auc)
print(accuracy)
