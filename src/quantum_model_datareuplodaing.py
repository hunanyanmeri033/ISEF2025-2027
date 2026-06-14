#
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train = np.load("../data/processed/X_trainq.npy")
y_train = np.load("../data/processed/y_trainq_tf.npy")

X_test = np.load("../data/processed/X_testq.npy")
y_test = np.load("../data/processed/y_testq_tf.npy")

X_valid = np.load("../data/processed/X_valq.npy")
y_valid = np.load("../data/processed/y_valq_tf.npy")
print("X_train")
print(X_train)
print("y_train----------------------------------------------------")
print(y_train)
print(X_train.shape)
print(y_train.shape)

print("X_test")
print(X_test)
y_train = 2 * y_train - 1
y_test  = 2 * y_test  - 1
y_valid = 2 * y_valid - 1
n_qubits = 6 # change to pca values
dev = qml.device("default.qubit", wires=n_qubits)


H = qml.Hamiltonian(
    coeffs=[1/n_qubits] * n_qubits,
    observables=[qml.PauliZ(i) for i in range(n_qubits)]
)


@qml.qnode(dev)
def feature_encoding(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4))
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
        qml.AngleEmbedding(features=x_chunks[i_layer % len(x_chunks)], wires=range(n_qubits), rotation='Y')
        for i, wire_params in enumerate(layer_params):
            qml.RX(wire_params, wires=i)
            #qml.RY(wire_params[1], wires=i)
            #qml.RZ(wire_params[2], wires=i)
        for s in range(n_qubits-1):
            
       # Entangle the qubits with CNOT gates
            qml.CNOT(wires=[s, s+1])
        #for i in range(n_qubits - 1):
        #   qml.CNOT(wires=[i, i+1])
    
    #print("---------------------------AAAAAAAAAAAAAAAAAAAAHHHHHHHHHHH------------")
    # Measure expectation value of Pauli-Z on the first qubit
    return qml.expval(H)


def loss(params, X, Y, lam=5e-4):
    preds = np.array([variational_circuit(params, x) for x in X])
    reg = lam * np.sum(params**2)
    #preds = np.tanh(5*preds)
    return np.mean(np.log(1 + np.exp(-Y * preds)))

def accuracy(params, X, Y):
    predictions = [variational_circuit(params, x) for x in X]
    predictions = np.sign(np.stack(predictions))
    return np.mean(predictions == Y)
#def F1
num_layers = 4
params = 0.01 * np.random.randn(num_layers, n_qubits, requires_grad=True)
print(params)
#opt = qml.GradientDescentOptimizer(stepsize=0.01)
opt = qml.AdamOptimizer(stepsize=0.03)
num_epochs = 300

epoch_list = []
loss_list = []
train_acc_list = []
test_acc_list = []
val_params = []
valid_acc_list = []

for epoch in range(num_epochs):
    # Update parameters by taking a gradient step on the loss w.r.t. params
    print("before .steo")
    old_params = params.copy()

    params = opt.step(loss, params, X=X_train, Y=y_train)
    print("after .step")
    current_loss = loss(params, X_train, y_train)
    train_acc = accuracy(params, X_train, y_train)
    test_acc = accuracy(params, X_test, y_test)
    valid_acc = accuracy(params, X_valid, y_valid)
    grad_fn = qml.grad(loss)
    g = grad_fn(params, X_train, y_train)
    #print("Gradient norm:", np.linalg.norm(g))

    preds = np.array([variational_circuit(params,x) for x in X_train])
    print("Pred range:", preds.min(), preds.max())
    print("Param change:", np.linalg.norm(params-old_params))
    epoch_list.append(epoch)
    loss_list.append(current_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    valid_acc_list.append(valid_acc)
    val_params.append(params)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {current_loss:.4f}, "
          f"Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}, val ACC = {valid_acc:.2f}")
biggest_acc = 0
for i in range(0,len(valid_acc_list)):
    if valid_acc_list[i]>biggest_acc:
        biggest_acc = valid_acc_list[i]
        val_index = i
plt.plot(epoch_list, loss_list, 'r--')
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('my_plot.png', dpi=300)
plt.show()

#metadata = {
#    "n_qubits": n_qubits,
#    "num_layers": num_layers,
#    "optimizer": "Adam",
#    "stepsize": 0.03,
#    "embedding": "AngleEmbedding-Y",
#    "entanglement": "ring",
#    "loss": "hinge"
#}`
#np.save("quantum_model_metadata.npy", metadata)
np.save("../models/quantum_params.npy", val_params[val_index])
print("Quantum model parameters and medadata saved.")
print("-------------------------Done-------------------------")
