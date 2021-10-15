import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tensorflow import keras
from preprocess import preprocess
import random
import matplotlib.pyplot as plt
from datetime import datetime
from inv_target import inv_target

# Input the path of the file where the optimized values and optional gifs will be saved
output_path = ''
gifs_path = ''

# DEFINE PATHS FOR LOADING ENERGY AND TEMPERATURE MODELS
ener_path = r''
temp_path = r''
ener_model = keras.models.load_model(f'{ener_path}')
temp_model = keras.models.load_model(f'{temp_path}')

# SET CORRECT SEQUENCE LENGTH BASED ON AFOREMENTIONED MODELS
SEQ_LEN = 4

# LOAD THE DATA USED IN OPTIMIZATION
data_path = ''
data = pd.read_csv(rf'{data_path}')

# HOW MANY ROUNDS TO OPTIMIZE?
opt_rounds = 2

# DEFINE THE BOUNDARIES OF OPTIMAL TEMPERATURE RANGE
lower_boundary = 21 # Ideal temperature range lower boundary in Celsius degrees
upper_boundary = 22 # Ideal temperature range upper boundary in Celsius degrees

print('OPTIMIZATION SCRIPT')
print('--------------------------------------------------------------------------------------------------------------------------------------------')

# Time the function running using datetime
start = datetime.now()

# Compile sequential data for energy consumption modeling
X_train, y_train, X_val, y_val, scaler = preprocess(raw_data=data, quant=['energy', 'temperature'], seq=SEQ_LEN, fut=1)

print(f'Shape of training data X {X_train.shape}, y {y_train.shape}')
print(f'Shape of validation data: X {X_val.shape}, y {y_val.shape}')

# Set cost function parameters, default values should work
low_temp = percentileofscore(data.temperature, lower_boundary)/100  # lower temperature boundary scaled
high_temp = percentileofscore(data.temperature, upper_boundary)/100 # upper temperature boundary scaled
print(f'Lower temperature boundary: {low_temp}')
print(f'Upper temperature boundary: {high_temp}')
p1 = 3 # Exponential Penalty for falling below ideal temperature range
p2 = 3 # Exponential Penalty for surpassing the ideal temperature range
cost1 = 10 # linear penalty coefficient for falling below the ideal temperature range
cost2 = 10 # Linear penalty coefficient for surpassing the ideal temperature range
n = 50  # Number of particles in optimization
N = 4   # Number of decision variables

optimized = np.empty((opt_rounds,2))

# Start the optimization loop
for val_point in range(opt_rounds):
    
    print(f'Begin optimization for point {val_point+1}')
    print('------------------------------------------------------------------------------------------------------------------------------------')
    
    # Initialize positions (x) and velocities (v) for n particles in N dimensions
    x = np.array([[random.random() for i in range(N)] for k in range(n)])
    v = np.array([[random.uniform(-0.1, 0.1) for i in range(N)] for k in range(n)])

    # Initialize each particles best known position (initial position) and whole swarms best know position
    l_hat = x   # Local best positions initialized as current positions
    inputs = np.array([np.concatenate((X_val[val_point,SEQ_LEN-1,:-N], x[k]), axis=None) for k in range(n)])  # Concatenate initial controls and non-optimized inputs
    inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))  # Reshape the data to include extra dimension for sequence
    inputs = np.array([np.concatenate((X_val[val_point,:-1], inputs[k]), axis=0) for k in range(n)]) # Concatenate the previous time instants to sequence
    
    # Make initial cost function values for all particles
    ener = np.array([ener_model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)])
    temp = np.array([temp_model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)])
    print(f'Initial energy predictions in round {val_point+1}:')
    print(ener)
    print(f'Initial temperature predictions in round {val_point+1}:')
    print(temp)
    low_penalty = np.array([(cost1*max(0, low_temp - temp[k]))**p1 for k in range(n)])
    print(f'Initial low. penalties in round {val_point+1}:')
    print(low_penalty)
    up_penalty = np.array([(cost2*max(0, temp[k] - high_temp))**p2 for k in range(n)])
    print(f'Initial up. penalties in round {val_point+1}:')
    print(up_penalty)
    outputs = ener + low_penalty + up_penalty # Combine different parts of cost function together
    print(f'Initial outputs in round {val_point+1}:')
    print(outputs)
    loc_best = outputs # Initiate local best vector with initial positions
    
    glob_best = min(loc_best)   # Find the initial global best cost function value
    g_hat = x[np.argmin(loc_best)] # Find the initial global best position 
    best_ener = ener[np.argmin(loc_best)]
    best_temp = temp[np.argmin(loc_best)]

    # Define values for inertia, cognitive and social parameters
    w = 0.5   # Stubbornness
    c1 = 0.3   # Tendency for attachment
    c2 = 0.3  # Ability for social learning 

    # Define a loop where the optimization process is run for k iterations
    iters = 10

    # Initiate a list for figure filenames
    names = []

    for i in range(iters):

        # Combine local best particle position with control data to form an input to sequential model
        inputs = np.array([np.concatenate((inputs[k,SEQ_LEN-1,:-N], x[k]), axis=None) for k in range(n)])
        inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))
        inputs = np.array([np.concatenate((X_val[val_point,:-1], inputs[k]), axis=0) for k in range(n)])
        
        # Use models to calculate cost function value for each particle position
        # Cost function consists of energy prediction and penalties for non-optimal room temperature values
        ener =  np.array([ener_model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)]) # Energy predictions in cost function
        temp = np.array([temp_model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)]) # Temperature predictions in cost function
        print(f'Energy predictions, round {val_point+1}, iter. {i+1}:')
        print(ener)
        print(f'Temperature predictions, round {val_point+1}, iter. {i+1}:')
        print(temp)
        
        low_penalty = np.array([(cost1*max(0, low_temp - temp[k]))**p1 for k in range(n)]) # Lower penalties in cost function
        print(f'Lower temperature penalties, round {val_point+1}, iter. {i+1}:')
        print(low_penalty)
        
        up_penalty = np.array([(cost2*max(0, temp[k] - high_temp))**p2 for k in range(n)]) # Upper penalties in cost function
        print(f'Upper temperature penalties, round {val_point+1}, iter. {i+1}:')
        print(up_penalty)
        
        outputs = ener + low_penalty + up_penalty # Combine previous values to cost function values
        
        # Check for each particle's best known position and update if applicable
        l_hat = np.array([x[k] if (outputs[k] < loc_best[k]) else l_hat[k] for k in range(n)])
        
        # Check and update the local best output values
        loc_best = [outputs[k] if (outputs[k] < loc_best[k]) else loc_best[k] for k in range(n)]
        
        # Check and update the global best output and position values
        if (min(loc_best) < glob_best):
            
            glob_best = min(loc_best)
            g_hat = x[np.argmin(loc_best)]
            best_ener = ener[np.argmin(loc_best)]
            best_temp = temp[np.argmin(loc_best)]
        
        print(f'Best energy consumption, round {val_point+1}, iter. {i+1}: {best_ener}')
        
        # Update particle velocities for all particles
        v = np.array([w*v[k] + c1*random.random()*(l_hat[k] - x[k]) + c2*random.random()*(g_hat - x[k]) for k in range(n)])
        
        # Update particle positions
        x = np.array([x[k] + v[k] for k in range(n)])
        
        # Check the position array for values over 1 and less than zero (optimization constraints)
        x = np.array([[0 if (x[j,k] < 0) else x[j,k] for k in range(N)] for j in range(n)])
        x = np.array([[1 if (x[j,k] > 1) else x[j,k] for k in range(N)] for j in range(n)])
    
    print('----------------------------------------')
    print('Optimized controls:')
    print(g_hat)
    
    print(f'Next sequence before adding optimized controls:')
    print(X_val[val_point+1])
    
    # Place optimized controls into correct position in future sequences
    for i in range(X_val.shape[1]-1):
        X_val[val_point+i+1,SEQ_LEN-i-2] = np.concatenate((X_val[val_point+i+1,SEQ_LEN-i-2,:-N], g_hat), axis=0)
        
    print(f'Next sequence after adding optimized controls:')
    print(X_val[val_point+1])
        
    # Store global best values for energy and temperature with optimal controls
    optimized[val_point,0] = best_ener
    optimized[val_point,1] = best_temp
    
    # Output to express progress in the script
    print(f'Best energy consumption value in round {val_point+1}: {best_ener}')
    print(f'Temperature after optimization of controls: {optimized[val_point,1]}')
    print(f'Best controls in round {val_point+1}: {g_hat}')
    print('---------------------------------------------------------------------------------------------------------------------------')

# Inverse transform the target values before storage
optimized, y_val = inv_target(X_val, optimized, y_val, scaler)
  
# Save the optimized and measured values to csv for later plotting/comparison
df = pd.DataFrame(data={'ener_opt': optimized[:,0], 'ener_meas': y_val[:,0], 'temp_opt': optimized[:,1], 'temp_meas': y_val[:,1]})

# Save the output
df.to_csv(output_path, index=False)

# How long did it take?
print('Script runtime:')
print(datetime.now() - start)