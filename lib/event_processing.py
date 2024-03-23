from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import jit, njit
from scipy.optimize import minimize

@njit
def blend_buffer(frames_buffer, mirror = False):
    ''' Get a frame buffer and compacts it into a single frame '''
    stacked_frames = np.zeros(frames_buffer[0].shape, dtype=np.uint16) 

    for i in range(frames_buffer.shape[0]):
        stacked_frames += frames_buffer[i]
    blend = stacked_frames / frames_buffer.shape[0] # Average the frames

    if mirror:
        blend = np.fliplr(blend)
    
    return blend.astype(np.uint8)  # Convert back to uint8 before returning


def calibration_blend(frames_buffer):
    ''' Get a frame buffer and compacts it into a single frame
    
    Parameters:
    ----------
        frames_buffer (list): list of frames to be compacted   
    Returns:
    -------
            blend (np.array): compacted frame

            '''
    
    blend = np.array(np.zeros((cv2.cvtColor(np.array(frames_buffer[0]), cv2.COLOR_BGR2GRAY).shape)))

    for frame in frames_buffer:
        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        blend += gray
    blend = blend

    return blend.astype(np.uint8)



def events_colapsing(evs, width, height):
    '''
    Create an array that colapse the events into an image
    
    Parameters
    ----------  
    evs : array
        Array of events
    width : int
        Width of the image
    height : int
        Height of the image

    Returns
    -------
    img : array
        Array of the image

    '''

    # Get the indices from ev and create an array with the pixel values
    img = np.full((width, height), 0)
    for ev in evs:
        if ev[2] == 0:
            img[ev[0], ev[1]] += 1

    return img

def treshold_filter(img, treshold):
    '''
    Deletes the pixels with a value below the treshold.

    Parameters
    ----------
    img : numpy array
        Image to be filtered.
    treshold : int
        Treshold value.

    Returns
    -------
    filtered_img : numpy array
        Filtered image.
    '''
    
    # Create a copy of the input image
    filtered_img = np.copy(img)

    # Filter the image with a threshold
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            if filtered_img[i, j] < treshold:
                filtered_img[i, j] = 0

    return filtered_img

def max_value_cluster(img, pixel_range, n_clusters):

    '''
    Creates a list of clusters searching the the maximun pixel value and performing the center of mass 
    of the pixels around it. The pixels that are part of the cluster are deleted and the process is repeated.

    Parameters
    ----------
    img : numpy array
        Image to be filtered.
    pixel_range : int
        Range of pixels to be considered around the maximun pixel value to perform the center of mass 
        (clustering).
    n_clusters : int
        Max number of clusters to be found.

    Returns
    -------
    clusters : list
        List of clusters. Each cluster is a list with the following structure:
            [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]    
    
    '''
    # Create a copy of the input image
    filtered_img = np.copy(img)

    clusters = []

    for i in range(n_clusters):
        # Get the maximun pixel value
        max_index = np.unravel_index(np.argmax(filtered_img), filtered_img.shape)

        cluster_x = 0
        cluster_y = 0
        cluster_mass = 0

        # Create a mask to select the pixels within the pixel range and check if the mask is out of the image
        mask = np.zeros_like(filtered_img, dtype=bool)
        mask[ (max_index[0] - pixel_range) if (max_index[0]-pixel_range) > 0 else 0:max_index[0] + pixel_range + 1 if 
             (max_index[0] + pixel_range + 1) < filtered_img.shape[0] else filtered_img.shape[0],
                (max_index[1] - pixel_range) if (max_index[1]-pixel_range) > 0 else 0:max_index[1] + pixel_range + 1 if
                (max_index[1] + pixel_range + 1) < filtered_img.shape[1] else filtered_img.shape[1]] = True

        # Get the coordinates and values of the pixels within the mask
        coords = np.argwhere(mask)
        values = filtered_img[mask]

        # Perform the clustering using vectorized operations
        cluster_x = np.sum(coords[:, 0] * values)
        cluster_y = np.sum(coords[:, 1] * values)
        cluster_mass = np.sum(values)

        # Calculate the cluster centroid
        if cluster_mass != 0:
            # Append the cluster to the list
            clusters.append([np.round([cluster_x, cluster_y] / cluster_mass).astype(int)
                          , cluster_mass, np.array(max_index)])

        # Set the pixels within the mask to zero 
        filtered_img[mask] = 0

    return clusters

def index_cluster(img, pixel_range, clusters_index):

    '''
    Creates a list of clusters performing the center of mass of the pixels around the index given.
    to it. The pixels that are part of the cluster are deleted and the process is repeated. For better
    perfomace the clusters should be ordered by the cluster mass.
    
    Parameters
    ----------
    img : numpy array
        Image to be filtered.
    pixel_range : int
        Range of pixels to be considered around the maximun pixel value to perform the center of mass 
        (clustering).
    
    clusters_index : list of arrays
        Array with the index of the clusters to be computed.
    

    Returns
    -------
    clusters : list
        List of clusters. Each cluster is a list with the following structure:
            [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]    
    
    '''
    # Create a copy of the input image
    filtered_img = np.copy(img)

    clusters = []

    for max_index in clusters_index:

        cluster_x = 0
        cluster_y = 0
        cluster_mass = 0

        # Create a mask to select the pixels within the pixel range and check if the mask is out of the image
        mask = np.zeros_like(filtered_img, dtype=bool)
        mask[ (max_index[0] - pixel_range) if (max_index[0]-pixel_range) > 0 else 0:max_index[0] + pixel_range + 1 if 
             (max_index[0] + pixel_range + 1) < filtered_img.shape[0] else filtered_img.shape[0],
                (max_index[1] - pixel_range) if (max_index[1]-pixel_range) > 0 else 0:max_index[1] + pixel_range + 1 if
                (max_index[1] + pixel_range + 1) < filtered_img.shape[1] else filtered_img.shape[1]] = True

        # Get the coordinates and values of the pixels within the mask
        coords = np.argwhere(mask)
        values = filtered_img[mask]

        # Perform the clustering using vectorized operations
        cluster_x = np.sum(coords[:, 0] * values)
        cluster_y = np.sum(coords[:, 1] * values)
        cluster_mass = np.sum(values)

        # Calculate the cluster centroid
        if cluster_mass != 0:
            # Append the cluster to the list
            clusters.append([np.round([cluster_x, cluster_y] / cluster_mass).astype(int)
                          , cluster_mass])

        # Set the pixels within the mask to zero
        filtered_img[mask] = 0

    return clusters

import cv2
import numpy as np


def order_by_center_dist(clusters, img_shape):
    '''
    Order the clusters by the distance to the center of the image.
    
    Parameters
    ----------
    clusters : list
        List of clusters. Each cluster is a list with the following structure:
            [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    img_shape : array
        Shape of the image.
    
    Returns
    -------
    clusters : list
        List of clusters ordered by the distance to the center of the image. Each cluster is a list with the following structure:
            [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    clusters_dist : array
            Array with the distance of each cluster to the center of the image.
    '''
    
    # New version: TODO: Check compatibility
    img_center = np.array(img_shape) / 2
    clusters_positions = np.array([cluster[0] for cluster in clusters])
    clusters_dist = np.linalg.norm(clusters_positions - img_center, axis=1)
    sorted_indices = np.argsort(clusters_dist)
    clusters = [clusters[i] for i in sorted_indices]
    clusters_dist = clusters_dist[sorted_indices]

    return clusters, clusters_dist

    img_center = np.array(img_shape)/2
    clusters = sorted(clusters, key=lambda x: np.linalg.norm(x[0] - img_center), reverse=False)

    #Compute a list of each cluster distance 
    clusters_dist = []
    for cluster in clusters:
        clusters_dist.append(np.linalg.norm(cluster[0] - img_center))
    clusters_dist = np.array(clusters_dist)

    return clusters, clusters_dist


def order_by_main_dist(main_cluster, clusters, get_index = False):
    '''
    Order the clusters by distance to the main cluster
    
    Parameters
    ----------
    main_cluster : list
        List with the following structure:
        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    clusters : list
        List of clusters. Each cluster is a list with the following structure:
        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
        
    Returns
    -------
    clusters : list
        List of clusters. Each cluster is a list with the following structure:
        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    '''
    
    if type(clusters[0][0]) == np.ndarray : # If the cluster contains also other info 
        sorted_clusters = sorted(clusters, key=lambda x: np.linalg.norm(x[0] - main_cluster[0]), reverse=False)

        #Get the index of the ordered clusters in the original list
        if get_index:
            sorted_clusters_index = []
            for id, cluster in enumerate([x[1] for x in sorted_clusters]):
                sorted_clusters_index.append([x[1] for x in clusters].index(cluster))
            return sorted_clusters, sorted_clusters_index
    
    elif type(clusters[0][0]) == np.int64 :# If the cluster is only cluster position [x_pixel, y_pixel]
        sorted_clusters = sorted(clusters, key=lambda x: np.linalg.norm(x - main_cluster), reverse=False)
        #Get the index of the ordered clusters in the original list
        if get_index:
            sorted_clusters_index = []
            for cluster in [x[1] for x in sorted_clusters]:
                sorted_clusters_index.append([x[1] for x in clusters].index(cluster))
            return sorted_clusters, sorted_clusters_index


    return sorted_clusters

def get_star_features(star_list, ref_pixel_to_deg = 1, reference_FOV = 1, recording_FOV = 1):
    ''' 
       Compute the star features for the first star of the list. The star list shoud be ordered by the 
       distance to the first star. 

       Parameters
       ----------
         star_list : list
              List of stars. Each star is a list with the following structure:
                [main_star[x_pixel, y_pixel], first_neirbour_star[x_pixel, y_pixel], second_neirbour_star[x_pixel, y_pixel], ...]
        
        Returns
        -------
        stars_features_1 : list
            List of the features 1 of each star. Each feature is a list with the following structure:
                [x_pixel, y_pixel]
        stars_features_2 : list
            List of the features 2 of each star. Each feature is a list with the following structure:
                [x_pixel, y_pixel]
    '''  
    star_features_1 = []
    star_features_2 = []
    pixel_to_deg = ref_pixel_to_deg * recording_FOV/reference_FOV
    star_list = np.array(star_list) * pixel_to_deg

#  Center the star list in the mean position (avoid traslation problems)
    star_list = star_list - np.mean(star_list, axis=0) #TODO: CHECK
    star_list = star_list - star_list[0]


    for j in range(len(star_list)):
        if j != 0:
            # Log polar transform
            star_features_1.extend(log_polar_transform(star_list[0],star_list[j], star_list[1]))
        # Compute distance between each neirbour star (permutation)
        for k in range(j+1,len(star_list)):
            star_features_2.append( np.linalg.norm(star_list[k] - star_list[j]) )
            # star_features_1.append( np.log(star_features_2[-1]) )


    star_features_1 = np.array(star_features_1) 
    star_features_2 = np.array(star_features_2)

    return star_features_1, star_features_2

def get_star_features_2(star_list,feature_type, ref_pixel_to_deg = 1, reference_FOV = 1, recording_FOV = 1):
    '''
    Compute the star features for the first star of the list. The star list shoud be ordered by the
    distance to the first star.

    PARAMETERS
    ----------
    star_list : list
        List of stars. Each star is a list with the following structure:
            [main_star[x_pixel, y_pixel], first_neirbour_star[x_pixel, y_pixel], second_neirbour_star[x_pixel, y_pixel], ...]
    feature_type : str
        Type of feature to be computed. Options: 'log_polar', 'permutation', 'permuatation_suffled', 'permutation_log', 'permutation_log_polar', 'permutation_angle', 'permutation_multi'

    '''  
    star_features_1 = []
    pixel_to_deg = ref_pixel_to_deg * recording_FOV/reference_FOV
    star_list = np.array(star_list) * pixel_to_deg

#  Center the star list in the mean position (avoid traslation problems)
    star_list = star_list - star_list[0]


    if feature_type == 'log_polar':

        for j in range(len(star_list)):
            if j == 1:
                star_features_1.extend(log_polar_transform(star_list[0],star_list[j], star_list[1])) 
                # star_features_1.pop()# Avoid angle =0 
                
            if j > 1:
                # Log polar transform
                star_features_1.extend(log_polar_transform(star_list[0],star_list[j], star_list[1]))

        # Compute distance between each neirbour star (permutation)
    elif feature_type == 'permutation':
    
        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.append( np.linalg.norm(star_list[k] - star_list[j]) )

    elif feature_type == 'permuatation_suffled':
    
        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.append( np.linalg.norm(star_list[k] - star_list[j]) )
            # flip the list
            star_features_1 = star_features_1[::-1]

    elif feature_type == 'permutation_log':
    
        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.append( np.log(np.linalg.norm(star_list[k] - star_list[j]) ))

    elif feature_type == 'permutation_log_polar':
    
        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.extend(log_polar_transform(star_list[j],star_list[k], star_list[1])) 
        star_features_1.pop(1)

    elif feature_type == 'permutation_angle':

        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.append( get_angle(star_list[j],star_list[k]))

    elif feature_type == 'permutation_multi':

        for j in range(len(star_list)):
            for k in range(j+1,len(star_list)):
                star_features_1.append( np.linalg.norm(star_list[k] - star_list[j]) * get_angle(star_list[j],star_list[k]))



    star_features_1 = np.array(star_features_1) 

    return star_features_1

def get_second(som, x):
    """Computes the coordinates of the winning neuron for the sample x."""
    activation_map = som.activate(x)
    # Get the second best matching unit
    return np.unravel_index(activation_map.argsort(axis=None)[1],
                             activation_map.shape)

def get_two_winners(som, x):
    """Computes the coordinates of the winning neuron for the sample x."""
    activation_map = som.activate(x)
    # Get the second best matching unit
    return np.unravel_index(activation_map.argsort(axis=None)[0],activation_map.shape), np.unravel_index(activation_map.argsort(axis=None)[1],activation_map.shape)

# Define a function to predict the star ID for a given feature vector
def predict_star_id(features, norm_param, dictionary, som, two_best_bmu = False):
    """
    Predict the star ID for a given feature vector.

    Parameters
    ----------
    features : array
        Feature vector.
    norm_param : array
        Array with the normalization parameters.
    dictionary : dict
        Dictionary with the star ID for each neuron.
    som : array
        Self Organizing Map.

    Returns
    -------
    star_id : int
        Star ID.
    """

    normalized_feature = (features - norm_param[0]) / (norm_param[1] - norm_param[0])

    if two_best_bmu == False:
        winner = som.winner(normalized_feature)
        if winner in dictionary:
            return dictionary[winner]
        else:
            return [0] #The neuron has no star ID return [0], the ID starts at 1 
    else:
        winner, second = get_two_winners(som, normalized_feature)
        if winner in dictionary:
            if second in dictionary:
                return dictionary[winner], dictionary[second]
            else:
                return dictionary[winner], [0] #The neuron has no star ID return [0], the ID starts at 1 
        else:
            if second in dictionary:
                return [0], dictionary[second]
            else:
                return [0], [0] #The neuron has no star ID return [0], the ID starts at 1 

def test_get_features(): 
    '''
    Test the get features function
    
    '''

    
    test_stars =[ [45.569912 , 4.089921],
                [45.593785 , 4.352873],
                [48.109873 , 6.660885],
                [49.839787 , 3.369980],
                [50.278217 , 3.675680]]
    test_stars = np.array(test_stars)

    test_sol_1 = [0.2640331480554887, 1.480255808333344, 3.6140362126900363, 0.7914639616175609, 4.330144122389013, -0.16703838162066847, 4.726492802567647, -0.08775497592500049]
    test_sol_2 = [0.2640331480554887, 3.6140362126900363, 4.330144122389013, 4.726492802567647, 3.4143256508227946, 4.358280776498913, 4.733127431736677, 3.717883378629331, 3.6896019300520404, 0.5344845768068048]
    features_1, features_2 = get_star_features(test_stars,1,1,1)


    print('Features 1 len: ',len(features_1), ', Features 2 len: ', len(features_2))
    print('Main star: ', test_stars[0]) 
    print('Features 1: ', features_1)
    print('Features 2: ', features_2)

    # for i in range(len(features_1)):
    #     print(f"Star posistion {test_stars[i]}  features: {features_1[i]}")
    #     print(f"    features 1: {features_1[i]}")
    #     print(f"    features 2: {features_2[i]}")

    print('Test error 1:', np.array(test_sol_1) - np.array(features_1))
    print('Test error 2:', np.array(test_sol_2) - np.array(features_2))


def log_polar_transform(main_point, secondary_point, axis_ref):
    ''' 
    Transforms x, y coordinates to log polar coordinates
    
    Parameters:
    x, y: float
        Cartesian coordinates
    x0, y0: float
        Center of the log polar coordinates (reference star)
        
    Returns:
    r, theta: float
        Log polar coordinates
        
    '''

    r = np.log( np.linalg.norm(main_point - secondary_point))
    theta = ( np.arctan2(secondary_point[1],secondary_point[0]) - np.arctan2(axis_ref[1],axis_ref[0]) + 2 * np.pi) % (2 * np.pi)
    # r = np.log(np.sqrt((x - x0)**2 + (y - y0)**2) )
    # theta = ( np.arctan2(y, x) - np.arctan2(y0, x0) + 2 * np.pi) % (2 * np.pi)

    return r, theta

def get_angle( secondary_point, axis_ref):
    ''' 
    Get the angle between two points with respect to the center

    Parameters:
    secondary_point : array
        Secondary star point
    axis_ref : array
        Axis reference point (usually closer to main star)

    Returns:
    theta : float
        Angle between the main point and the secondary point with respect to the axis reference
        
    '''
    theta = ( np.arctan2(secondary_point[1],secondary_point[0]) - np.arctan2(axis_ref[1],axis_ref[0]) + 2 * np.pi) % (2 * np.pi)

    return theta

def distance(x_c, y_c, points):
    '''Calculate the distance of the points to the center of the image'''
    return np.sqrt((points[:, 0] - x_c)**2 + (points[:, 1] - y_c)**2)

def objective(point_c, *args):
    ''' Objective function to minimize the distance of the points to the center of the image'''
    points, distances = args
    x_c, y_c = point_c
    return np.sum((distance(x_c, y_c, points) - distances)**2)

def solve_point_c(points, distances):
    '''Solve the center of the image'''
    initial_guess = np.mean(points, axis=0)  # Initial guess for [x_c, y_c]
    result = minimize(objective, initial_guess, args=(points, distances))
    return result.x


def check_star_id_by_neight( indices_neigh_gt, indices_neigh_image, indices_image, extend_puzzle = False):
    ''' Check the star id by comparing the neighbours of the star and the dataset neighbours.

    Parameters
    ----------
    indices_neigh_gt : np.array
        List of the neighbours of the stars in the ground truth.
    indices_neigh_image : np.array
        List of the neighbours of the stars in the dataset.
    indices_image : np.array
        List of the stars in the dataset.
    extend_puzzle : bool
        If True, the function returns the star id with the neighbours. If False, the function returns the star id without the neighbours.

    Returns
    -------
    confirmed_stars_ids : np.array
        List of the confirmed stars ids.

    '''

    check_points = np.zeros((len(indices_neigh_gt)))

    for i in range(len(indices_neigh_gt)): # For all clustes that are not None
        if indices_neigh_gt[i][0] is not None:
            for j in range(1,len(indices_neigh_image[i])): # For all the neighbours of the cluster [1,5]
                if indices_neigh_image[i][j] == indices_neigh_gt[i][j]: # If neighbours MATCH +1 for the star and the neighbour 
                    check_points[i] += 2
                    check_points[indices_image[i][j]] += 2
                elif indices_neigh_image[i][j] is not None: # If neighbours DONT MATCH -1 for the star and the neighbour 
                    check_points[i] -= 1
                    check_points[indices_image[i][j]] -= 1
                    break

    confirmed_indices = [i for i in range(len(check_points)) if check_points[i] > 0] # Index of original list of confrimed stars
    confirmed_stars_ids = np.full(len(indices_neigh_gt), None)

    if extend_puzzle:
        for index in confirmed_indices:
            confirmed_stars_ids[indices_image[index].tolist()] =  indices_neigh_gt[index] # WITH IDS NOT HIP NUMBER 
    else:
        for index in confirmed_indices:
            confirmed_stars_ids[index] = indices_neigh_gt[index][0] # WITH IDS NOT HIP NUMBER 

    return confirmed_stars_ids, confirmed_indices