import numpy as np
from scipy.special import lambertw
from pysr import pysr, best

def inverse_function(x):
    return -lambertw(2.718281828459045**(x-3))+x-3

# Generate data for W(x)
x_values = np.linspace(0.1, 10, 100)  # Generate input values for x (avoid x=0 because of singularity)
y_values = np.vectorize(inverse_function)(x_values)  # Compute y = W(x) for each x

# Fit a symbolic regression model
equations = pysr(x_values, y_values, niterations=1000, binary_operators=["+", "*", "/", "-"], unary_operators=["exp", "log"])

# View the best equation found
print(best(equations))
