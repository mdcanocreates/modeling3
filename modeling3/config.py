"""
Configuration settings for Modeling 3 pipeline.
"""

import numpy as np
from pathlib import Path

# Random seed for reproducibility
RANDOM_SEED = 42

# Image size for normalization (all images resized to this)
IMAGE_SIZE = (256, 256)

# Channel order and mapping
# Original channels: BF, Actin, Microtubules, Nuclei
# Internal aliases: phase (BF), actin, mt (Microtubules), nuc (Nuclei)
CHANNEL_ORDER = ["BF", "Actin", "Microtubules", "Nuclei"]
CHANNEL_ALIASES = {
    "BF": "phase",
    "Actin": "actin",
    "Microtubules": "mt",
    "Nuclei": "nuc"
}

# Data root directory
DATA_ROOT = Path("img_model")

# Output directory
OUTPUT_ROOT = Path("modeling3_outputs")

# Generation parameters
N_IMAGES_PER_ALGORITHM = 15  # Minimum per algorithm (≥15)
N_IMAGES_PER_CELL = 10  # Per algorithm per cell (3 cells × 10 = 30 total per alg)

# Algorithm 1 (Classical Augmentation) parameters
ALG1_ROTATION_RANGE = (-15, 15)  # Degrees
ALG1_ELASTIC_ALPHA = 50  # Elastic deformation strength
ALG1_ELASTIC_SIGMA = 5  # Elastic deformation smoothness
ALG1_NOISE_STD = 0.02  # Gaussian noise standard deviation
ALG1_BRIGHTNESS_RANGE = (0.8, 1.2)  # Brightness jitter
ALG1_CONTRAST_RANGE = (0.8, 1.2)  # Contrast jitter
ALG1_ZOOM_RANGE = (0.9, 1.1)  # Zoom/crop range

# Algorithm 2 (TPS Warping) parameters
ALG2_N_CONTROL_POINTS = 10  # Number of control points for TPS
ALG2_DISPLACEMENT_MAX = 20  # Maximum displacement in pixels

# Algorithm Prime (VAE) parameters
ALG_PRIME_LATENT_DIM = 64  # Latent space dimension (reduced for small dataset)
ALG_PRIME_BATCH_SIZE = 4  # Training batch size
ALG_PRIME_N_EPOCHS = 80  # Number of training epochs
ALG_PRIME_LEARNING_RATE = 1e-4  # Learning rate for Adam optimizer
ALG_PRIME_RECON_WEIGHT = 1.0  # Weight for reconstruction loss
ALG_PRIME_KL_WEIGHT = 1e-6  # Weight for KL divergence regularization (very small, essentially autoencoder-like)
ALG_PRIME_TRAIN_SPLIT = 0.8  # Fraction of data for training (rest for validation)

# Algorithm D (Diffusion) parameters
# IMPORTANT: This is a scaffold for future work only. Algorithm D is NOT used in any reported results.
# Set DIFFUSION_ENABLED = True to enable (disabled by default for safety).
DIFFUSION_ENABLED = False  # Master flag: must be True for any diffusion code to run
DIFFUSION_NUM_STEPS = 1000  # Number of diffusion steps (forward and reverse)
DIFFUSION_BETA_START = 0.0001  # Starting noise schedule value
DIFFUSION_BETA_END = 0.02  # Ending noise schedule value
# Note: DIFFUSION_IMAGE_SIZE reuses IMAGE_SIZE (256, 256) - no separate config needed

# Clustering parameters
N_CLUSTERS = 3  # K-Means k value

# Figure parameters
FIGURE_DPI = 300  # Publication-ready DPI
FIGURE_FORMAT = "png"  # or "pdf"

# Logging
LOG_FILE = "errors_mc3.log"

