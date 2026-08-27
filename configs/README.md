# Experiment configuration

The current three-day MVP exposes configuration through explicit command-line arguments. This directory is reserved for versioned experiment configurations when the number of runs grows. Defaults are centralized in `src/styleseek/paths.py`, while run-specific hyperparameters are stored automatically in training history and Ultralytics run metadata.
