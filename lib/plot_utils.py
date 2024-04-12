from matplotlib import axis
from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt

import cv2

from mpl_toolkits.mplot3d import Axes3D

def plot_image(img):
    # Create figure and axes 
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    fig.set_size_inches(18.5, 10.5)
    #delete axis
    ax.axis('off')

    plt.show()

    

    # Pause for 0.01 seconds
    plt.pause(0.01)

def plot_cluster(img, clusters, cluster_size, size = [10, 7]):
    '''
    Plot the clusters on the image. 

        Parameters
        ----------
        img : numpy array
            Image to be filtered.
        clusters : list
                List of clusters. Each cluster is a list with the following structure:
                        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
        cluster_size : int
                Size of the square to be plotted around the cluster.
        size : list 
                Size of the image.

        Returns
        -------
        None.
    '''
    # Create figure and axes 
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img, cmap='gray')

    # If the clusters are in the full cluster form [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    if len(clusters[0]) > 2:

        # Plot a square around the initial clusters position with size cluster_size
        for cluster in clusters:
                ax.plot([cluster[2][1] - cluster_size, cluster[2][1] - cluster_size, cluster[2][1] + cluster_size, cluster[2][1] + cluster_size, cluster[2][1] - cluster_size],
                        [cluster[2][0] - cluster_size, cluster[2][0] + cluster_size, cluster[2][0] + cluster_size, cluster[2][0] - cluster_size, cluster[2][0] - cluster_size], 'r')

        #     # Plot a square around the clusters with size cluster_size 
        #     for cluster in clusters:
        #         ax.plot([cluster[0][1] - cluster_size, cluster[0][1] - cluster_size, cluster[0][1] + cluster_size, cluster[0][1] + cluster_size, cluster[0][1] - cluster_size],
        #                 [cluster[0][0] - cluster_size, cluster[0][0] + cluster_size, cluster[0][0] + cluster_size, cluster[0][0] - cluster_size, cluster[0][0] - cluster_size], 'g', linestyle='dotted')
                
        # # Plot the cluster number as text in the top left corner of the cluster
        #         for i, cluster in enumerate(clusters):
        #                 ax.text(cluster[2][1] - cluster_size, cluster[2][0] - cluster_size -5, i+1, color='r')

        # Plot a red cross on the center of mass of the clusters
                ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                        [cluster[0][0] - 2, cluster[0][0] + 2], 'g')
                ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                        [cluster[0][0] + 2, cluster[0][0] - 2], 'g')

        # Plot a red cross on the initial position of the clusters
                ax.plot([cluster[2][1] - 2, cluster[2][1] + 2],
                        [cluster[2][0] - 2, cluster[2][0] + 2], 'r')
                ax.plot([cluster[2][1] - 2, cluster[2][1] + 2],
                        [cluster[2][0] + 2, cluster[2][0] - 2], 'r')
                
    # If cluster is only cluster position [x_pixel, y_pixel]
    elif len(clusters[0]) == 2:
        for cluster in clusters:
                # Plot a square around the initial clusters position with size cluster_size
                ax.plot([cluster[1] - cluster_size, cluster[1] - cluster_size, cluster[1] + cluster_size, cluster[1] + cluster_size, cluster[1] - cluster_size],
                        [cluster[0] - cluster_size, cluster[0] + cluster_size, cluster[0] + cluster_size, cluster[0] - cluster_size, cluster[0] - cluster_size], 'r')
                
                # Plot a red cross on the initial position of the clusters
                ax.plot([cluster[1] - 2, cluster[1] + 2],
                        [cluster[0] - 2, cluster[0] + 2], 'r')
                ax.plot([cluster[1] - 2, cluster[1] + 2],
                        [cluster[0] + 2, cluster[0] - 2], 'r')
                # Plot the cluster number as text in the top left corner of the cluster
                # for i, cluster in enumerate(clusters):
                #         ax.text(cluster[1] - cluster_size, cluster[0] - cluster_size -5, i+1, color='r')
    fig.set_size_inches(size[0], size[1])

#     Crop the image to the origina size
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    # Axis off
    ax.axis('off')
    # Show the plot
    plt.show()

def plot_cluster_with_id(img, clusters, ids, size = [10, 7], center = False):
    '''
    Plot the clusters on the image. 

        Parameters
        ----------
        img : numpy array
            Image to be filtered.
        clusters : list
                List of clusters. Each cluster is a list with the following structure:
                        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
        ids : list
                ids of identified stars.
        size : list 
                Size of the image.

        Returns
        -------
        None.
    '''
    # Create figure and axes 
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img, cmap='gray')

    # If the clusters are in the full cluster form [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
    if len(clusters[0]) > 2:
        print('Error: clusters in full form not implemented yet')           
    # If cluster is only cluster position [x_pixel, y_pixel]
    elif len(clusters[0]) == 2:
        for cluster in clusters:
                # Plot a red cross on the initial position of the clusters
                ax.plot([cluster[1] - 2, cluster[1] + 2],
                        [cluster[0] - 2, cluster[0] + 2], 'r')
                ax.plot([cluster[1] - 2, cluster[1] + 2],
                        [cluster[0] + 2, cluster[0] - 2], 'r')
                # Plot the cluster number as text in the top left corner of the cluster
                for i, cluster in enumerate(clusters):
                        ax.text(cluster[1] - 5, cluster[0]  -5, ids[i], color='r')
    
    fig.set_size_inches(size[0], size[1])

        #Plot the center of the image and the axis
    if center:
        ax.plot([img.shape[1]//2, img.shape[1]//2],
                [0, img.shape[0]], 'g')
        ax.plot([0, img.shape[1]],
                [img.shape[0]//2, img.shape[0]//2], 'g')

    # Axis off
    ax.axis('off')
    # Show the plot
    plt.show()

def plot_cluster_cv(frame, clusters_list,confirmed_stars_ids,original_confirmed_ids,frame_position = None, size = None, show_confirmed_ids = False):
        '''
        Plot the clusters on the image. 

        size : list 
            Size of the image.

        Returns
        -------
        None.
        '''
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        cluster_size = 15
        clusters =  clusters_list


        for i, cluster in enumerate(clusters):
                
            # Plot a square around the initial clusters position with size cluster_size
            cv2.rectangle(img_rgb, (cluster[1] - cluster_size, cluster[0] - cluster_size),
                        (cluster[1] + cluster_size, cluster[0] + cluster_size), (0, 0, 255), 1)
            # Plot a red cross on the initial position of the clusters
            cv2.drawMarker(img_rgb, (cluster[1], cluster[0]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5)

            # Plot the cluster number as text in the top left corner of the cluster

            if show_confirmed_ids and  confirmed_stars_ids is not None:

                # Full confirmed stars  stars (light green)
                if confirmed_stars_ids[i] is not None:

                    if confirmed_stars_ids[i]  not in original_confirmed_ids: # Full confirmed stars (light green)
                        cv2.putText(img_rgb, str(confirmed_stars_ids[i]), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        # Update the square and the cross to green if the star is confirmed
                        cv2.rectangle(img_rgb, (cluster[1] - cluster_size, cluster[0] - cluster_size),
                            (cluster[1] + cluster_size, cluster[0] + cluster_size), (255, 255, 0), 1)
                        cv2.drawMarker(img_rgb, (cluster[1], cluster[0]), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=5)
                        
                    else: # Original confirmed stars (green)
                        cv2.putText(img_rgb, str(confirmed_stars_ids[i]), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # Update the square and the cross to green if the star is confirmed
                        cv2.rectangle(img_rgb, (cluster[1] - cluster_size, cluster[0] - cluster_size),
                            (cluster[1] + cluster_size, cluster[0] + cluster_size), (0, 255, 0), 1)
                        cv2.drawMarker(img_rgb, (cluster[1], cluster[0]), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=5)
                        
            else:
                for i, cluster in enumerate(clusters):
                    cv2.putText(img_rgb, str(i+1), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        

        if frame_position is not None:
            cv2.putText(img_rgb, 'Frame position: ' + str(frame_position), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # # # Resize image
        if size is not None:
            img_rgb_resized = cv2.resize(img_rgb, (size[0], size[1]))
            return img_rgb_resized
        else:       
            return img_rgb
        
def plot_cluster_comparation(img, clusters_batchs, cluster_size):
    '''
    Plot the clusters on the image. 

        Parameters
        ----------
        img : numpy array
            Image to be filtered.
        clusters_batchs : list
                List of clusters. Each cluster is a list with the following structure:
                        [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]
        cluster_size : int
                Size of the square to be plotted around the cluster.

        Returns
        -------
        None.

    '''
    # Create figure and axes 
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img, cmap='gray')
    
    spacing = 0 # Text spacing 

    # Plot a square around the initial clusters position with size cluster_size
    for clusters in clusters_batchs:
        
        for cluster in clusters:
            # ax.plot([cluster[2][1] - cluster_size, cluster[2][1] - cluster_size, cluster[2][1] + cluster_size, cluster[2][1] + cluster_size, cluster[2][1] - cluster_size],
            #         [cluster[2][0] - cluster_size, cluster[2][0] + cluster_size, cluster[2][0] + cluster_size, cluster[2][0] - cluster_size, cluster[2][0] - cluster_size], 'r')
    
        # Plot the cluster number as text in the top left corner of the cluster
            for i, cluster in enumerate(clusters):
                    ax.text(cluster[2][1] - cluster_size +spacing, cluster[2][0] - cluster_size , i+1, color='r')

        # Plot a red green on the center of mass of the clusters
        for cluster in clusters:
            ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                    [cluster[0][0] - 2, cluster[0][0] + 2],)
            ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                    [cluster[0][0] + 2, cluster[0][0] - 2],)
            
        spacing += 25

    ax.set_title('Comparison of the clustering methods')
    ax.axis('off')

    # Show the plot
    plt.show()


def cv_plot(img):
    cv2.imshow('image',img)  
    cv2.waitKey() # This is necessary to be required so that the image doesn't close immediately.   
    cv2.destroyAllWindows()



def plot_sphere_with_trajectory(radius=1, center=(0, 0, 0), num_samples=100, trajectory=None):
    # Generate theta and phi values
    theta = np.linspace(0, 2 * np.pi, num_samples)
    phi = np.linspace(0, np.pi, num_samples)

    # Generate coordinates for the sphere surface
    x = radius * np.outer(np.cos(theta), np.sin(phi)) + center[0]
    y = radius * np.outer(np.sin(theta), np.sin(phi)) + center[1]
    z = radius * np.outer(np.ones(np.size(theta)), np.cos(phi)) + center[2]

    # Plot the sphere surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.6)

    # Set aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Plot the trajectory if provided
    if trajectory is not None:
        latitudes, longitudes = trajectory
        # Convert latitude and longitude to radians
        latitudes_rad = np.radians(latitudes)
        longitudes_rad = np.radians(longitudes)
        
        # Convert latitude a
