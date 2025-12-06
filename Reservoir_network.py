import numpy as np
import pandas as pd


# 1. Initialize Parameters

N_in = 3           
N_res = 500        
N_out = 3          
N_predict = 500    

in_variance = 0.002
res_variance = 0.004
in_std = np.sqrt(in_variance)
res_std = np.sqrt(res_variance)

ridge_param = 0.01

# Target spectral radius for reservoir weight matrix
spectral_radius_target = 0.95

# Washout periods for training and testing [steps to discard]
washout_train = 100
washout_test = 100 

# Set random seed for reproducibility
np.random.seed(42)


# 2. Load Data

try:
    train_df = pd.read_csv('training-set.csv', header=None)
    test_df = pd.read_csv('test-set-8.csv', header=None)

    # assignemnt expects data in shape (T, 3)
    train_data = train_df.values.T
    test_data = test_df.values.T
    
    if train_data.shape[1] != N_in:
        if train_df.values.shape[1] == N_in:
             train_data = train_df.values
             test_data = test_df.values

    
    T_train = train_data.shape[0]
    T_test = test_data.shape[0]

    print(f"Training data loaded: {train_data.shape}")
    print(f"Test data loaded: {test_data.shape}")

except FileNotFoundError:
    print("Error: 'training-set.csv' or 'test-set-8.csv' not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 3. Initialize Weights

W_in = np.random.normal(0, in_std, (N_res, N_in))
W_res = np.random.normal(0, res_std, (N_res, N_res))

# Scale W_res to have the desired spectral radius
eigvals = np.linalg.eigvals(W_res)
spectral_radius = np.max(np.abs(eigvals))

if spectral_radius == 0:
    print("Warning: Spectral radius is 0.")
else:
    W_res *= (spectral_radius_target / spectral_radius)
    print(f"Spectral radius scaled from {spectral_radius:.4f} to {spectral_radius_target}.")

print("Weight matrices W_in and W_res initialized.")


# 4. Training Phase

# Matrix to store all reservoir states
X_states = np.zeros((T_train, N_res))
x_t = np.zeros((N_res, 1)) # Initial state

if T_train <= washout_train:
    raise ValueError("Training data is too short for washout period.")

for t in range(T_train - 1):
    u_t = train_data[t].reshape(-1, 1)
    x_t = np.tanh(W_in @ u_t + W_res @ x_t)
    X_states[t+1] = x_t.flatten()

# Discard the washout period
print(f"Discarding first {washout_train} washout steps...")
X_design = X_states[washout_train:]
Y_target = train_data[washout_train:]

print(f"X_design shape: {X_design.shape}")
print(f"Y_target shape: {Y_target.shape}")

# Calculate W_out using Ridge Regression
# Solves: W_out.T = inv(X.T @ X + k*I) @ X.T @ Y

I = np.identity(N_res)
inv_term = np.linalg.inv(X_design.T @ X_design + ridge_param * I)
W_out_T = inv_term @ X_design.T @ Y_target

# W_out shape is (N_out, N_res)
W_out = W_out_T.T

print(f"W_out calculated, shape: {W_out.shape}")


# 5. Prediction Phase: Autonomous generation

predictions = np.zeros((N_predict, N_out))

if T_test < washout_test:
    raise ValueError("Test data is too short for priming.")

# Use test data to get reservoir into the right state
print(f"Priming reservoir with {washout_test} test steps...")
x_t = np.zeros((N_res, 1)) # Reset state
for t in range(washout_test):
    u_t = test_data[t].reshape(-1, 1)
    x_t = np.tanh(W_in @ u_t + W_res @ x_t)

# Autonomous Generation
current_x = x_t
predicted_u = W_out @ current_x # First prediction
predictions[0] = predicted_u.flatten()

print(f"Generating {N_predict} autonomous steps...")
for i in range(N_predict - 1):
    # Use last prediction as next input
    current_u = predicted_u 
    
    # Update state
    current_x = np.tanh(W_in @ current_u + W_res @ current_x)
    
    # Get new prediction
    predicted_u = W_out @ current_x
    
    # Store prediction
    predictions[i+1] = predicted_u.flatten()

print("Prediction complete.")

# 6. Save Results

y_predictions = predictions[:, 1]

output_filename = 'prediction.csv'

print(f"Predicted y-component saved to {output_filename}")