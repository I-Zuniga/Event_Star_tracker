from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt
import cv2


def blend_buffer(frames_buffer):
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
    blend = blend/len(frames_buffer)

    return blend.astype(np.uint8)


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

def max_value_cluster(img, pixel_range, n_clusters, iterations=1):

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
        Number of clusters to be found.
    iterations : int, optional
        Number of times the clustering is performed. When iterating the clusters are ordered by the cluster 
        mass and the index used a preliminary stimations . The default is 1.

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
        # Performs the clustering on the pixel range around the maximun pixel value
        for i in range(-pixel_range, pixel_range):
            for j in range(-pixel_range, pixel_range):
                if max_index[0]+i < filtered_img.shape[0] and max_index[1]+j < filtered_img.shape[1]:
                    
                    cluster_x += filtered_img[max_index[0] + i , max_index[1] + j ] * i
                    cluster_y += filtered_img[max_index[0] + i , max_index[1] + j ] * j
                    cluster_mass += filtered_img[max_index[0] + i , max_index[1] + j ] 

                    # Delete the pixels that are part of the cluster to find another maximun
                    filtered_img[max_index[0] + i , max_index[1] + j ] = 0

        #CHECK 
        # plot_image(filtered_img)
        if cluster_mass != 0:
            clusters.append([ np.round(max_index + [cluster_x, cluster_y]/cluster_mass).astype(int), 
                            cluster_mass, 
                            np.array(max_index)]) 

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

    for k in range(len(clusters_index)):
        # Get the maximun pixel value
        cluster_index = clusters_index[k]

        cluster_x = 0
        cluster_y = 0
        cluster_mass = 0
        # Performs the clustering on the pixel range around the maximun pixel value
        for i in range(-pixel_range, pixel_range):
            for j in range(-pixel_range, pixel_range):
                if cluster_index[0]+i < filtered_img.shape[0] and cluster_index[1]+j < filtered_img.shape[1]:

                    cluster_x += filtered_img[cluster_index[0] + i , cluster_index[1] + j ] * i
                    cluster_y += filtered_img[cluster_index[0] + i , cluster_index[1] + j ] * j
                    cluster_mass += filtered_img[cluster_index[0] + i , cluster_index[1] + j ] 

                    # Delete the pixels that are part of the cluster to find another maximun
                    filtered_img[cluster_index[0] + i , cluster_index[1] + j ] = 0

        #CHECK 
        # plot_image(filtered_img)
        
        if cluster_mass != 0:
           clusters.append([ np.round(cluster_index + [cluster_x, cluster_y]/cluster_mass).astype(int), 
                            cluster_mass, 
                            np.array(cluster_index)]) 

    return clusters

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
    img_center = np.array(img_shape)/2
    clusters = sorted(clusters, key=lambda x: np.linalg.norm(x[0] - img_center), reverse=False)

    #Compute a list of each cluster distance 
    clusters_dist = []
    for cluster in clusters:
        clusters_dist.append(np.linalg.norm(cluster[0] - img_center))
    clusters_dist = np.array(clusters_dist)

    return clusters, clusters_dist


def order_by_main_dist(main_cluster, clusters):
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
    
    # Order the clusters by distance to the main cluster
    clusters = sorted(clusters, key=lambda x: np.linalg.norm(x[0] - main_cluster[0]), reverse=False)

    return clusters

def get_star_features(star_list):
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

    # Compute the x and y distance between the each star and the rest of the stars
    for j in range(1,len(star_list)):
            # x and y distance between the stars
            star_features_1.append(star_list[j][0] - star_list[0][0])
            star_features_1.append(star_list[j][1] - star_list[0][1])

    for j in range(len(star_list)):
        # Compute distance btwen each neirbour star (permutation)
        for k in range(j+1,len(star_list)):
            star_features_2.append( np.linalg.norm(star_list[k] - star_list[j]) )


    return star_features_1, star_features_2


# Define a function to predict the star ID for a given feature vector
def predict_star_id(features, norm_param, dictionary, som):
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
    winner = som.winner(normalized_feature)
    if winner in dictionary:
        return dictionary[winner]
    else:
        return [0] #The neuron has no star ID return [0], the ID start at 1 