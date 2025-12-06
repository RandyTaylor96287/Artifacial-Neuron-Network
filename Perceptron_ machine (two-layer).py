import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Define two-layer Neural Network 
# ---------------------------------------------------
class NeuralNetwork:

    def __init__(self, M1, M2, learning_rate=0.01):
 
        self.M1 = M1
        self.M2 = M2
        self.eta = learning_rate

        # initialize weights and thresholds
        # use small random values to avoid vanshing gradients
        # first layer
        self.w1 = np.random.randn(self.M1, 2) * 0.1
        self.theta1 = np.random.randn(self.M1, 1) * 0.1   

        # second layer
        self.w2 = np.random.randn(self.M2, self.M1) * 0.1
        self.theta2 = np.random.randn(self.M2, 1) * 0.1

        # output layer
        self.w3 = np.random.randn(self.M2, 1) * 0.1
        self.theta3 = np.random.randn(1, 1) * 0.1

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2

    def forward_pass(self, x):

        # permutate x to be a column vector
        x = x.reshape(-1, 1)

        # first layer
        # h1 is sum
        self.h1 = np.dot(self.w1, x) - self.theta1
        self.V1 = self._tanh(self.h1)

        # second layer
        self.h2 = np.dot(self.w2, self.V1) - self.theta2
        self.V2 = self._tanh(self.h2)

        # output layer
        self.h3 = np.dot(self.w3.T, self.V2) - self.theta3
        O = self._tanh(self.h3)
        
        return O, self.V1, self.V2

    def backward_pass(self, x, t, O):
        x = x.reshape(-1, 1)

        # compute output layer error of back propogation 
        delta3 = (O - t) * self._tanh_derivative(self.h3)

        # compute second layer error of back propogation
        delta2 = np.dot(self.w3, delta3) * self._tanh_derivative(self.h2)

        # compute first layer error of back propogation
        delta1 = np.dot(self.w2.T, delta2) * self._tanh_derivative(self.h1)

        # compute delta gradients
        # dE/dw3 = dE/dh3 * dh3/dw3 = delta3 * V2
        grad_w3 = self.V2 * delta3
        grad_theta3 = -delta3

        # dE/dw2 = dE/dh2 * dh2/dw2 = delta2 * V1.T
        grad_w2 = np.dot(delta2, self.V1.T)
        grad_theta2 = -delta2

        # dE/dw1 = dE/dh1 * dh1/dw1 = delta1 * x.T
        grad_w1 = np.dot(delta1, x.T)
        grad_theta1 = -delta1
        
        return grad_w1, grad_theta1, grad_w2, grad_theta2, grad_w3, grad_theta3

    def update_weights(self, grad_w1, grad_theta1, grad_w2, grad_theta2, grad_w3, grad_theta3):

        self.w1 = self.w1 - self.eta * grad_w1
        self.theta1 = self.theta1 - self.eta * grad_theta1
        
        self.w2 = self.w2 - self.eta * grad_w2
        self.theta2 = self.theta2 - self.eta * grad_theta2
        
        self.w3 = self.w3 - self.eta * grad_w3
        self.theta3 = self.theta3 - self.eta * grad_theta3

        return 0

    def train_sequential(self, x_train, t_train):

        # forward propogation
        O, _, _ = self.forward_pass(x_train) # feed output value only
        
        # backward propogation
        grads = self.backward_pass(x_train, t_train, O)
        
        # iteration
        self.update_weights(*grads)

        return 0

# ---------------------------------------------------
# 2. Define the evaluation function
# ---------------------------------------------------
def calculate_classification_error(model, x_val, t_val):

    p_val = len(x_val)
    if p_val == 0:  # in case of empty validation set
        return 0
        
    error_sum = 0
    for i in range(p_val):
        x_sample = x_val[i]  # avoid modifying the original data
        t_sample = t_val[i]
        
        O, _, _ = model.forward_pass(x_sample)
        O_temp  = O[0]
        sgn_O = np.sign(O_temp)
        
        error_sum += np.abs(sgn_O - t_sample)
    C = error_sum / (2 * p_val)
    return C.item()

# ---------------------------------------------------
# 3. Revoke data
# ---------------------------------------------------
# if/else indicator in pandas library version
try: 
    # upload training set
    train_df = pd.read_csv('training_set (1).csv', header=None)  # straightforwardly get data
    X_train = train_df.iloc[:, [0, 1]].values   # read data according to the number of columns and rows
    T_train = train_df.iloc[:, 2].values

    # upload validation set
    val_df = pd.read_csv('validation_set (1).csv', header=None)
    X_val = val_df.iloc[:, [0, 1]].values
    T_val = val_df.iloc[:, 2].values
    
    print(f"traning set upload sucessfully: {len(X_train)} samples")
    print(f"validation set upload sucessfully: {len(X_val)} samples")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("please check 'training_set (1).csv' and 'validation_set (1).csv' are in the SCRIPT DIRECTORY.")
    exit()


# ---------------------------------------------------
# 4. Setup learning parameters and start training
# ---------------------------------------------------
# initialize parameters
M1 = 55  # the number of neurons in the first hidden layer
M2 = 55  # the number of neurons in the second hidden layer
epochs = 200
learning_rate = 0.01

model = NeuralNetwork(M1=M1, M2=M2, learning_rate=learning_rate)

validation_errors = []

print("\nAlgorithm: Stochastic Gradient Descent (SGD)")
print(f"Network structure: 2 -> {M1} -> {M2} -> 1")
print(f"Epochs: {epochs}")
print(f"Learning Rate: {learning_rate}")

for epoch in range(epochs):

    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    T_train_shuffled = T_train[permutation]  # Fancy indexing of numpy (it just works)
    
    for i in range(len(X_train_shuffled)):
        x_sample = X_train_shuffled[i]
        t_sample = T_train_shuffled[i]
        model.train_sequential(x_sample, t_sample)  # revoke function call from perceptron machine
        
    current_error = calculate_classification_error(model, X_val, T_val)
    validation_errors.append(current_error)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Validation Classification Error (C): {current_error:.4f}")

print("Training complete!")


try:
    np.savetxt("w1.csv", model.w1, delimiter=",")
    np.savetxt("w2.csv", model.w2, delimiter=",")
    np.savetxt("w3.csv", model.w3, delimiter=",")
    np.savetxt("t1.csv", model.theta1, delimiter=",")
    np.savetxt("t2.csv", model.theta2, delimiter=",")
    np.savetxt("t3.csv", model.theta3, delimiter=",")
    print("CSV files saved successfully.")
except Exception as e:
    print(f"Error when trying to save CSV files: {e}")


# ---------------------------------------------------
# 5. Visualization
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), validation_errors, marker='o', linestyle='-')
plt.title('Validation Classification Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Classification Error (C)')
plt.grid(True)
plt.show()
final_error = validation_errors[-1]
print(f"\n Final error rate (C): {final_error:.4f}")