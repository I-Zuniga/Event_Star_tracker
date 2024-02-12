from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt

import cv2


def plot_image(img):
    # Create figure and axes 
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    fig.set_size_inches(18.5, 10.5)
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
                
        # Plot the cluster number as text in the top left corner of the cluster
                for i, cluster in enumerate(clusters):
                        ax.text(cluster[2][1] - cluster_size, cluster[2][0] - cluster_size -5, i+1, color='r')

        # Plot a red green on the center of mass of the clusters
        for cluster in clusters:
                ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                        [cluster[0][0] - 2, cluster[0][0] + 2], 'g')
                ax.plot([cluster[0][1] - 2, cluster[0][1] + 2],
                        [cluster[0][0] + 2, cluster[0][0] - 2], 'g')

        # Plot a red cross on the initial position of the clusters
        for cluster in clusters:
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
                # Plot the cluster number as text in the top left corner of the cluster
                for i, cluster in enumerate(clusters):
                        ax.text(cluster[1] - cluster_size, cluster[0] - cluster_size -5, i+1, color='r')
    fig.set_size_inches(size[0], size[1])

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