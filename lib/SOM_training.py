import os
path = os.getcwd()
# Get parent directory
parent = os.path.dirname(path)
#Add parent directory to system path
os.sys.path.insert(0, parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import sys
import platform
from scipy.spatial import KDTree

from lib.utils import *
from lib.event_processing import *

# Define a function to predict the star ID for a given feature vector
def predict_star_id(features, features_array, dictionary, som):
    normalized_feature = (features - features_array.min(axis=0)) / (features_array.max(axis=0) - features_array.min(axis=0))
    winner = som.winner(normalized_feature)
    if winner in dictionary:
        return dictionary[winner]
    else:
        return [0] #The neuron has no star ID return [0], the ID start at 1 
    
# Normalize the data -> better performace of the SOM 
def normalize_features(star_features):
    star_features_normalized = (star_features - star_features.min(axis=0)) / (star_features.max(axis=0) - star_features.min(axis=0))
    return star_features_normalized

# Another dict for map neurons to star (just to check), same as star_ids but created from som.winner rather than som.winner_map
def add_values_in_dict(sample_dict, key, list_of_values):
    ''' Append multiple values to a key in 
        the given dictionary '''
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict


def train(hyperparameters):
    ## Select the dataset type: 'random' or 'tycho'
    # stars_data = utils.get_star_dataset(type ='random', n_stars = 4000)
    catalog_path = '../data/catalogs/tycho2_VT_6.csv'
    stars_data = get_star_dataset(type ='tycho', path = catalog_path)

    # If data type is a dataframe from a catalog transform it to array
    if isinstance(stars_data, pd.DataFrame):
        # stars_data['data_number'] = stars_data.index
        stars_data = stars_data[['HIP','RA(ICRS)', 'DE(ICRS)']].values

    # Create the k-d tree to find the nearest neighborhoods of the center stars
    # As this is used only for the training of the SOM performace is not needed
    tree = KDTree(stars_data[:,1:3])

    n_of_neighbor = 4 # Number of neighborhoods stars used to compute the features

    # Find the 5 closest neighbors for each star
    distances, indices = tree.query(stars_data[:,1:3], k=n_of_neighbor+1)

    # Create the k-d tree to find the nearest neighborhoods of the center stars
    # As this is used only for the training of the SOM performace is not needed
    tree = KDTree(stars_data[:,1:3])

    n_of_neighbor = 4 # Number of neighborhoods stars used to compute the features

    # Find the 5 closest neighbors for each star
    distances, indices = tree.query(stars_data[:,1:3], k=n_of_neighbor+1)

    #Initialice the dual SOM features vector
    features_vec_1 = []
    features_vec_2 = []

    # Compute the distances in the x and y axes to each of the five closest stars for each star
    for i in range(len(stars_data)):

        # Initialice the subsets of features
        features_1 = []
        features_2 = []

        for j in range(1,n_of_neighbor+1):
            neighbor_index = indices[i][j]
            x_distance = stars_data[neighbor_index][1] - stars_data[i][1]
            y_distance = stars_data[neighbor_index][2] - stars_data[i][2]

            #  Define the features vector that is going to be used in the SOM:
            features_1.extend(log_polar_transform(stars_data[i][1], stars_data[i][2], stars_data[neighbor_index][1], stars_data[neighbor_index][2]) )
                
            for k in range(1 +j-1,n_of_neighbor+1):
                features_2.append( np.sqrt( (stars_data[indices[i][k]][1] - stars_data[indices[i][j-1]][1])**2
                                +(stars_data[indices[i][k]][2] - stars_data[indices[i][j-1]][2])**2 )
                                ) 

        features_vec_1.append(features_1)
        features_vec_2.append(features_2)


    ## SINGLE TRAINING ##

    features_vec_1 = np.array(features_vec_1)
    features_vec_2 = np.array(features_vec_2)

    features_1_n = normalize_features(np.array(features_vec_1))
    features_2_n = normalize_features(np.array(features_vec_2))

    # Must be more neurons that points -> sqrt of data size to set the mesh: data < som_rows * som_cols
    # mesh_size = int(np.sqrt(features_1_n.shape[0])/2)
    # Manual set 

    # Initialize the SOM
    som1 = MiniSom(
        x = hyperparameters['mesh_size_1'],
        y = hyperparameters['mesh_size_1'],
        input_len = features_1_n.shape[1],

        sigma=hyperparameters['sigma'],
        learning_rate=hyperparameters['learning_rate'],
        neighborhood_function=hyperparameters['neighborhood_function'],
        topology=hyperparameters['topology'],
        activation_distance=hyperparameters['activation_distance']

    )

    som2 = MiniSom(
        x = hyperparameters['mesh_size_2'],
        y = hyperparameters['mesh_size_2'],
        input_len = features_2_n.shape[1],

        sigma=hyperparameters['sigma'],
        learning_rate=hyperparameters['learning_rate'],
        neighborhood_function=hyperparameters['neighborhood_function'],
        topology=hyperparameters['topology'],
        activation_distance=hyperparameters['activation_distance']
    )

    # Train the SOM
    som1.train_random(data=features_1_n, num_iteration=50000)
    som2.train_random(data=features_2_n, num_iteration=50000)

    som1.quantization_error

    star_dict_1= {}
    star_dict_2= {}

    for i in range(len(features_1_n)):
        star_dict_1 = add_values_in_dict(star_dict_1, som1.winner(features_1_n[i]),[i])
        star_dict_2 = add_values_in_dict(star_dict_2, som2.winner(features_2_n[i]),[i])

    cont = np.zeros(3) # [Correct match, miss match, multiple match]
    mean_noise = 0
    for i in range(features_vec_1.shape[0]):

        # Itroduce noise in the features vector to check the response of the SOM
        scale = 0.005 # % respect max value 

        noise_1 = np.random.normal(loc=0, scale=1, size=features_vec_1.shape[1])*np.max(features_vec_1[i])*scale
        noise_2 = np.random.normal(loc=0, scale=1, size=features_vec_2.shape[1])*np.max(features_vec_2[i])*scale
        sample_feature_1= features_vec_1[i] - noise_1
        sample_feature_2= features_vec_2[i] - noise_2

        mean_noise += np.mean( np.abs(noise_1)/features_vec_1[i] ) + np.mean( np.abs(noise_2)/features_vec_2[i] ) 

        predicted_star_ids_1 = predict_star_id(sample_feature_1,np.array(features_vec_1),star_dict_1,som1)
        predicted_star_ids_2 = predict_star_id(sample_feature_2,np.array(features_vec_2),star_dict_2,som2)

        star_guess = list(set(predicted_star_ids_1).intersection(predicted_star_ids_2))
        if len(list(set(predicted_star_ids_1).intersection(predicted_star_ids_2))) == 1:
            cont[0] += star_guess[0] == i
        else:
            # print("Error: ", list(set(predicted_star_ids_1).intersection(predicted_star_ids_2)), "!=", i)
            cont[1] += len(star_guess) == 0
            cont[2] += len(star_guess) > 1
    mean_noise = mean_noise / features_vec_1.shape[0] / 2

    return cont[0] / features_vec_1.shape[0]

# if __name__ == "__main__":
#     hyperparameters = {
#         'sigma': 1,
#         'learning_rate': 0.2,
#         'neighborhood_function': 'gaussian',
#         'topology': 'hexagonal',
#         'activation_distance': 'euclidean'
#     }
#     accuracy = train(hyperparameters)
#     print(accuracy)