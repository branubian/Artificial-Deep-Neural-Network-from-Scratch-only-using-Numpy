import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv("3-spiral.csv")  ## reads file stored in same folder as this code & makes it a dataframe.
m = np.array(d)
##print(m, "this is data matrix and it's shape is", m.shape)

##plt.show()

X = m[:, :2]
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) ## standard scaling the data
##print(X, "this is input matrix and it's shape is", X.shape)
Y = m[:, 2]
Y = np.array(Y).reshape(-1, 1)
# only use if dataset classes are 1,2,3 instead of 0,1,2
Y[Y == 1] = 0
Y[Y == 2] = 1
Y[Y == 3] = 2
##print(Y, "this is labels matrix and it's shape is", Y.shape)


def one_hot(y):
    y_int = y.astype(int)
    num_classes = len(np.unique(y_int))
    one_hot_encoded = np.eye(num_classes)[y_int.reshape(-1)]
    return one_hot_encoded


Y_o_h = one_hot(Y)

##print(one_hot(Y), "this is labels matrix one hot and it's shape is", one_hot(Y).shape)

n1 = int(input("enter number of neurons in first hidden layer: "))
n2 = int(input("enter number of neurons in second hidden layer: "))
n3 = int(input("enter number of neurons in third hidden layer: "))
n4 = int(input("enter number of neurons in fourth hidden layer: "))
n5 = int(input("enter number of neurons in fifth hidden layer: "))
alpha = float(input("enter floating point learning rate for gradient descent: "))
epochs = int(input("enter integer value for number of training epochs: "))
n = [2, n1, n2, n3, n4, n5, 3]
np.random.seed(0)
W1 = 0.01 * np.random.randn(n[0], n[1])
W2 = 0.01 * np.random.randn(n[1], n[2])
W3 = 0.01 * np.random.randn(n[2], n[3])
W4 = 0.01 * np.random.randn(n[3], n[4])
W5 = 0.01 * np.random.randn(n[4], n[5])
W6 = 0.01 * np.random.randn(n[5], n[6])
b1 = np.random.randn(1, n[1])
b2 = np.random.randn(1, n[2])
b3 = np.random.randn(1, n[3])
b4 = np.random.randn(1, n[4])
b5 = np.random.randn(1, n[5])
b6 = np.random.randn(1, n[6])


def softmax(inputs):
    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    output = probabilities
    return output


def loss(pred, o_h_true):  ## categorical cross entropy loss
    pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
    true_clipped = np.clip(o_h_true, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(pred_clipped * true_clipped, axis=1)
    negative_logs = -np.log(correct_confidences)
    loss = np.mean(negative_logs)
    return loss


def forward():
    global Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6, Y_hat, W1, W2, W3, W4, W5, W6, b1, b2, b3, b4, b5, b6, Loss
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = np.maximum(0, Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = np.maximum(0, Z4)
    Z5 = np.dot(A4, W5) + b5
    A5 = np.maximum(0, Z5)
    Z6 = np.dot(A5, W6) + b6
    Y_hat = softmax(Z6)  ## A6
    Loss=loss(Y_hat, Y_o_h)
    ##print(Y_hat, "this is Y_hat, it's dimensions are:", Y_hat.shape)
    ##print(Loss, "this is the categorical cross entropy loss")

def backward():
    global dL_dW6, dL_db6, dL_dW5, dL_db5, dL_dW4, dL_db4, dL_dW3, dL_db3, dL_dW2, dL_db2, dL_dW1, dL_db1
    samples = len(Y_hat) # counts number of rows of Y_hat
    # Copy so we can safely modify
    Y_1D=Y.ravel()
    dL_dZ6 = Y_hat.copy()
    # Calculate gradient
    dL_dZ6[range(samples), Y_1D.astype(int)] -= 1
    dL_dZ6 = dL_dZ6 / samples
    dL_dW6 = np.dot(A5.T, dL_dZ6)
    dL_db6 = np.sum(dL_dZ6, axis=0, keepdims=True)
    dL_dA5 = np.dot(dL_dZ6, W6.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ5 = dL_dA5.copy()
    # Zero gradient where input values were negative
    dL_dZ5[Z5 <= 0] = 0
    dL_dW5 = np.dot(A4.T, dL_dZ5)
    dL_db5 = np.sum(dL_dZ5, axis=0, keepdims=True)
    dL_dA4 = np.dot(dL_dZ5, W5.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ4 = dL_dA4.copy()
    # Zero gradient where input values were negative
    dL_dZ4[Z4 <= 0] = 0
    dL_dW4 = np.dot(A3.T, dL_dZ4)
    dL_db4 = np.sum(dL_dZ4, axis=0, keepdims=True)
    dL_dA3 = np.dot(dL_dZ4, W4.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ4 = dL_dA4.copy()
    # Zero gradient where input values were negative
    dL_dZ4[Z4 <= 0] = 0
    dL_dW4 = np.dot(A3.T, dL_dZ4)
    dL_db4 = np.sum(dL_dZ4, axis=0, keepdims=True)
    dL_dA3 = np.dot(dL_dZ4, W4.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ3 = dL_dA3.copy()
    # Zero gradient where input values were negative
    dL_dZ3[Z3 <= 0] = 0
    dL_dW3 = np.dot(A2.T, dL_dZ3)
    dL_db3 = np.sum(dL_dZ3, axis=0, keepdims=True)
    dL_dA2 = np.dot(dL_dZ3, W3.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ2 = dL_dA2.copy()
    # Zero gradient where input values were negative
    dL_dZ2[Z2 <= 0] = 0
    dL_dW2 = np.dot(A1.T, dL_dZ2)
    dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)
    dL_dA1 = np.dot(dL_dZ2, W2.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ1 = dL_dA1.copy()
    # Zero gradient where input values were negative
    dL_dZ1[Z1 <= 0] = 0
    dL_dW1 = np.dot(X.T, dL_dZ1)
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)

def update_params_vanilla():
    global W1, W2, W3, W4, W5, W6, b1, b2, b3, b4, b5, b6
    W1 = W1 - (alpha * dL_dW1)
    W2 = W2 - (alpha * dL_dW2)
    W3 = W3 - (alpha * dL_dW3)
    W4 = W4 - (alpha * dL_dW4)
    W5 = W5 - (alpha * dL_dW5)
    W6 = W6 - (alpha * dL_dW6)
    b1 = b1 - (alpha * dL_db1)
    b2 = b2 - (alpha * dL_db2)
    b3 = b3 - (alpha * dL_db3)
    b4 = b4 - (alpha * dL_db4)
    b5 = b5 - (alpha * dL_db5)
    b6 = b6 - (alpha * dL_db6)

def update_params_ADAM():
    # Hyperparameters
    # learning rate
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 1  # time step (increment each iteration)

    # --- Parameters ---
    W = {
        'W1': W1, 'W2': W2, 'W3': W3,
        'W4': W4, 'W5': W5, 'W6': W6
    }
    b = {
        'b1': b1, 'b2': b2, 'b3': b3,
        'b4': b4, 'b5': b5, 'b6': b6
    }

    # --- Gradients ---
    dW = {
        'W1': dL_dW1, 'W2': dL_dW2, 'W3': dL_dW3,
        'W4': dL_dW4, 'W5': dL_dW5, 'W6': dL_dW6
    }
    db = {
        'b1': dL_db1, 'b2': dL_db2, 'b3': dL_db3,
        'b4': dL_db4, 'b5': dL_db5, 'b6': dL_db6
    }

    # --- Moment estimates (initialize to zeros only once) ---
    mW = {k: np.zeros_like(v) for k, v in W.items()}
    vW = {k: np.zeros_like(v) for k, v in W.items()}
    mb = {k: np.zeros_like(v) for k, v in b.items()}
    vb = {k: np.zeros_like(v) for k, v in b.items()}

    # --- Adam Update ---
    for k in W:
        # Update weights
        mW[k] = beta1 * mW[k] + (1 - beta1) * dW[k]
        vW[k] = beta2 * vW[k] + (1 - beta2) * (dW[k] ** 2)

        mW_hat = mW[k] / (1 - beta1 ** t)
        vW_hat = vW[k] / (1 - beta2 ** t)

        W[k] -= alpha * mW_hat / (np.sqrt(vW_hat) + epsilon)

    for k in b:
        # Update biases
        mb[k] = beta1 * mb[k] + (1 - beta1) * db[k]
        vb[k] = beta2 * vb[k] + (1 - beta2) * (db[k] ** 2)

        mb_hat = mb[k] / (1 - beta1 ** t)
        vb_hat = vb[k] / (1 - beta2 ** t)

        b[k] -= alpha * mb_hat / (np.sqrt(vb_hat) + epsilon)

    # Increment timestep
    t += 1

def get_accuracy():
    global accuracy, predictions
    ##print(f"{Y_hat} This is softmax_outputs, it's dimensions are {Y_hat.shape}")
    # Target (ground-truth) labels for 3 samples
    class_targets = Y.ravel()
    class_targets = class_targets.astype(int)

    ##print(f"{class_targets} This is class_targets, it's dimensions are {class_targets.shape}")
    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(Y_hat, axis=1)

    ##print(f"{predictions} this is prediction matrix, it's dimensions: {predictions.shape}")

    accuracy = np.mean(predictions == class_targets)
    ##print(f"accuracy over training data is {accuracy*100}%")

epoch_count = []
loss_count = []
accuracy_count = []

for i in range(1,epochs+1):
    forward()
    get_accuracy()
    accuracy_count.append(float(accuracy*100))
    loss_count.append(float(Loss))
    backward()
    update_params_ADAM()
    epoch_count.append(int(i))
    print(i,"epochs completed")




print(f"final loss is {Loss}")
print(f"final accuracy is {accuracy*100}%")
plt.subplot(2,2,1)

##print(f"{loss_count}, this is all losses")
##print(f"{epoch_count}, this is all epochs")
# Create a 2x2 grid layout
##plt.figure(figsize=(8, 4))

# First plot spans the top row (2 columns)
##ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
plt.plot(epoch_count,loss_count)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"final loss is {Loss}")

plt.subplot(2,2,2)
plt.plot(epoch_count,accuracy_count)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (in %age)")
plt.title(f"final accuracy is {accuracy*100}%")

plt.subplot(2,2,3)
plt.scatter(m[:, 0], m[:, 1], c=m[:, 2], cmap='brg')
plt.title("Data for Training")

plt.subplot(2,2,4)
plt.scatter(x=m[:, 0],y=m[:, 1], c=predictions, cmap='brg')
plt.title("How the ANN classifies it")

plt.tight_layout() ##prevents overwriting in graphs
plt.show()

print(dL_dW1)
print(dL_dW2)
print(dL_dW3)

