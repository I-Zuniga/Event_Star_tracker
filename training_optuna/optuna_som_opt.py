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

    # SOM 1 
    mesh_size1_x = trial.suggest_int('mesh_size1_x', 60,90)
    mesh_size1_y = trial.suggest_int('mesh_size1_y', 60,90)
    sigma_1 = trial.suggest_float('sigma_1', 2, 5)
    learning_rate_1 = trial.suggest_float('learning_rate_1', 0.6, 2)
    neighborhood_function_1 = trial.suggest_categorical('neighborhood_function_1', ['gaussian', 'triangle'])
    topology_1 = trial.suggest_categorical('topology_1', ['hexagonal', 'rectangular'])
    activation_distance_1 = trial.suggest_categorical('activation_distance_1', ['euclidean', 'manhattan'])
    feature_type_1 = trial.suggest_categorical('feature_type_1', ['log_polar', 'permutation', 'permutation_log', 'permutation_log_polar', 'permutation_angle', 'permutation_angle_dist'])

    mesh_size2_x = trial.suggest_int('mesh_size2_x', 60,90)
    mesh_size2_y = trial.suggest_int('mesh_size2_y', 60,90)
    sigma_2 = trial.suggest_float('sigma_2', 2, 5)
    learning_rate_2 = trial.suggest_float('learning_rate_2', 0.6, 2)
    neighborhood_function_2 = trial.suggest_categorical('neighborhood_function_2', ['gaussian', 'triangle'])
    topology_2 = trial.suggest_categorical('topology_2', ['hexagonal', 'rectangular'])
    activation_distance_2 = trial.suggest_categorical('activation_distance_2', ['euclidean', 'manhattan'])
    feature_type_2 = trial.suggest_categorical('feature_type_2', ['log_polar', 'permutation', 'permutation_log', 'permutation_log_polar','permutation_angle', 'permutation_angle_dist'])



    hyperparameters_1 = {
        'mesh_size_x': mesh_size1_x,
        'mesh_size_y': mesh_size1_y,
        'sigma': sigma_1,
        'learning_rate': learning_rate_1,
        'neighborhood_function': neighborhood_function_1,
        'topology': topology_1,
        'activation_distance': activation_distance_1,
        'n_of_neighbors' : 4,
        'epochs' : 100000,
        'feature_type': feature_type_1,
    }

    hyperparameters_2 = {
        'mesh_size_x': mesh_size2_x,
        'mesh_size_y': mesh_size2_y,
        'sigma': sigma_2,
        'learning_rate': learning_rate_2,
        'neighborhood_function': neighborhood_function_2,
        'topology': topology_2,
        'activation_distance': activation_distance_2,
        'n_of_neighbors' : 4,
        'epochs' : 100000,
        'feature_type': feature_type_2,
    }

    accuracy = train(hyperparameters_1, hyperparameters_2, trial)
   
    
    return accuracy


# if __name__ == "_main_":
    
print("Study statistics: ")

study = optuna.create_study(
    # direction='maximize', # single objective
    directions=['maximize', 'maximize','maximize'], # multi-objective
    storage="sqlite:///db.sqlite3",
    study_name="Multi Objective SOM Optimization",
    load_if_exists=True,
    pruner=optuna.pruners.ThresholdPruner(lower=0.1),
    )
study.optimize(objective, n_trials=200)  # You can adjust the number of trials


pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

# Print the best hyperparameters
#print('Best hyperparameters:', study.best_params)

# Retrieve the best hyperparameters
#best_hyperparams = study.best_params

# Create an instance of your TD3 model with the best hyperparameters
#run(best_hyperparams, final=True)