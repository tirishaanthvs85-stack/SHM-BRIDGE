"""
Configuration File
Bridge SHM PINN
"""

# Beam Length (m)
LENGTH = 10.0

# Beam Length (m)
L = 10.0

# Young's Modulus (Pa)
E = 2.1e11

# Moment of Inertia (m^4)
I = 8.33e-6

# Uniform Load (N/m)
Q = 1000

# Number of Training Points
N_DATA = 200

# Physics Collocation Points
N_PHYSICS = 500

# Training Epochs
EPOCHS = 5000

# Learning Rate
LEARNING_RATE = 0.001

# Hidden Layers
HIDDEN_LAYERS = 5

# Neurons per Layer
NEURONS = 50

# Random Seed
SEED = 42

# Save Model Path
MODEL_PATH = "models/bridge_pinn.keras"

# Results Folder
RESULT_FOLDER = "results/"

LAMBDA_DATA = 1.0
LAMBDA_PHYSICS = 10.0
LAMBDA_BOUNDARY = 100.0