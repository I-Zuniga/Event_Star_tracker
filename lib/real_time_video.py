
from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt

import cv2

from utils import *
from lib.event_processing import *

class ClusterVideo:
    def __init__(self, wait_time = 100, size = None):
        self.size = size
        self.wait_time = wait_time

        self.new_frame = False

    def update_frame(self, frame):
        self.frame = frame
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(self.wait_time) == ord('q'):
            print('Exit key pressed')
            return True
    
class ClusterFrame: 
    def __init__(self, frame, index_clustering = True, mass_treshold = None, treshold_filter = 0.2, pixel_range = 15):
        self.frame = frame

        # Parameters & options 
        self.treshold = treshold_filter # Percetange of max value 
        self.pixel_range = pixel_range
        self.index_clustering = index_clustering
        self.mass_treshold = mass_treshold

        self.clusters_list = []

    def compute_clusters(self):

        frame = cv2.blur(self.frame,(3,3))
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        max_val = np.max(frame)
        frame_thr = cv2.threshold(frame, max_val*self.treshold, 1, cv2.THRESH_TOZERO)[1]
        
        clusters =  max_value_cluster(frame_thr, self.pixel_range, 30)
        clusters = sorted(clusters, key=lambda x: x[1], reverse=True)
        clusters_index = np.array([cluster[0] for cluster in clusters])
        clusters_index = sorted(clusters_index, key=lambda x: x[1], reverse=True)

        if self.index_clustering:
            self.clusters_list = index_cluster(frame, self.pixel_range, clusters_index)

        if self.mass_treshold is not None:
            treshold_val = self.mass_treshold*np.max([cluster_mass[1] for cluster_mass in  self.clusters_list])
            self.clusters_list = [cluster for cluster in self.clusters_list if cluster[1] > treshold_val ]

        return self.clusters_list
    
    def update_clusters(self, frame):
        self.frame = frame
        self.compute_clusters()


    def plot_cluster(self, size = [10, 7]):
        '''
        Plot the clusters on the image. 

            Parameters
            ----------
            img : numpy array
                Image to be filtered.
            clusters : list
                    List of clusters. Each cluster is a list with the following structure:
                            [ [x_pixel, y_pixel], cluster comulative mass, inital cluster position [x_max, y_max] ]

            size : list 
                    Size of the image.

            Returns
            -------
            None.
        '''

        cluster_size = self.pixel_range
        img = self.frame
        clusters =  [x[0] for x in self.clusters_list]

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
                    
        # If cluster is only cluster position [x_pixel, y_pixel]
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

    def plot_cluster_cv(self, size = None):
        '''
        Plot the clusters on the image. 

        size : list 
            Size of the image.

        Returns
        -------
        None.
        '''
        img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)

        cluster_size = self.pixel_range
        clusters =  [x[0] for x in self.clusters_list]

        # # If cluster is only cluster position [x_pixel, y_pixel]

        for cluster in clusters:
                # Plot a square around the initial clusters position with size cluster_size
                cv2.rectangle(img_rgb, (cluster[1] - cluster_size, cluster[0] - cluster_size),
                            (cluster[1] + cluster_size, cluster[0] + cluster_size), (0, 0, 255), 1)

                # Plot a red cross on the initial position of the clusters
                cv2.drawMarker(img_rgb, (cluster[1], cluster[0]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5)

                # Plot the cluster number as text in the top left corner of the cluster
                for i, cluster in enumerate(clusters):
                    cv2.putText(img_rgb, str(i+1), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # # # Resize image
        if size is not None:
            img_rgb_resized = cv2.resize(img_rgb, (size[0], size[1]))
            return img_rgb_resized
        else:       
            return img_rgb

