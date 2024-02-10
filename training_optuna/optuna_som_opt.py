# Get system path 
import os
path = os.getcwd()
# Get parent directory
parent = os.path.dirname(path)
#Add parent directory to system path
os.sys.path.insert(0, parent)

import optuna
import minisom
import numpy as np
from scipy.spatial import KDTree

# from optuna.pruners import BasePruner
# from optuna.trial._state import TrialState

from lib.som_training import *

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# from stable_baselines import PPO2
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.cmd_util import make_vec_env

# # https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb
# from custom_env import GoLeftEnv


def objective(trial):
    # Number of layers and units per layer for actor
    mesh_size_1 = trial.suggest_int('mesh_size_1', 30,67)
    mesh_size_2 = trial.suggest_int('mesh_size_2', 30,67)
    # mesh_size_1 = 40
    # mesh_size_2 = 40

    sigma = trial.suggest_float('sigma', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.2, 2)

    neighborhood_function = trial.suggest_categorical('neighborhood_function', ['gaussian', 'bubble', 'triangle'])
    topology = trial.suggest_categorical('topology', ['hexagonal', 'rectangular'])
    activation_distance = trial.suggest_categorical('activation_distance', ['euclidean', 'manhattan'])
    
    # n_of_neighbors = trial.suggest_int('n_of_neighbors', 3,6)


    hyperparameters = {
        'mesh_size_1': mesh_size_1,
        'mesh_size_2': mesh_size_2,
        'sigma': sigma,
        'learning_rate': learning_rate,
        'neighborhood_function': neighborhood_function,
        'topology': topology,
        'activation_distance': activation_distance,
        'n_of_neighbors' : 4,
    }

    accuracy = train(hyperparameters)

    # Handle pruning based on the intermediate value.    
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()
    
    return accuracy


# if __name__ == "_main_":
    
print("Study statistics: ")

study = optuna.create_study(
    direction='maximize', # single objective
    # directions=['maximize', 'minimize'], # multi-objective
    storage="sqlite:///db.sqlite3",
    study_name="features_log_polar_euclidean_4_neighbors",
    load_if_exists=True,
    )
study.optimize(objective, n_trials=1000)  # You can adjust the number of trials


# pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

# Print the best hyperparameters
#print('Best hyperparameters:', study.best_params)

# Retrieve the best hyperparameters
#best_hyperparams = study.best_params

# Create an instance of your TD3 model with the best hyperparameters
#run(best_hyperparams, final=True)