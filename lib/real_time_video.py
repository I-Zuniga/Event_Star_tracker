
from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import minisom

from utils import *
from lib.utils import *
from lib.plot_utils import *   
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

        # Parameters & options for filtering
        self.treshold = treshold_filter # Percetange of max value 
        self.pixel_range = pixel_range # Pixel range of the cluster
        self.index_clustering = index_clustering # If true, second iteration of clustering (reduce duplicates but slower)
        self.mass_treshold = mass_treshold # Optional: Treshold for the mass of the clusters (if not None)

        # Parameters for pixel to degree conversion
        self.ref_pixel_to_deg = 0.0009601418439716312 #In degres from sun_calibration with FOV=reference_FOV
        self.reference_FOV = 1 #In degrees
        self.recording_FOV = 14.3 #In degrees
        self.num_of_neirbours = 4

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

        self.clusters_list, _= order_by_center_dist(self.clusters_list, self.frame.shape)

    
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

    def plot_cluster_cv(self, size = None, show_con_ids = False):
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

                if show_con_ids and self.confirmed_stars_ids is not None:
                    confirmed_stars_hip = [int(self.stars_data[x][0]) if x is not None else None for x in self.confirmed_stars_ids]
                    for i, cluster in enumerate(clusters):
                        cv2.putText(img_rgb, str(confirmed_stars_hip[i]), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    for i, cluster in enumerate(clusters):
                        cv2.putText(img_rgb, str(i+1), (cluster[1] - cluster_size, cluster[0] - cluster_size - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # # # Resize image
        if size is not None:
            img_rgb_resized = cv2.resize(img_rgb, (size[0], size[1]))
            return img_rgb_resized
        else:       
            return img_rgb
    
    def load_star_catalog(self, path = '../data/catalogs/tycho2_VT_6.csv'):
        stars_data_df = get_star_dataset(type ='tycho', path = path)
        self.stars_data = stars_data_df[['HIP','RA(ICRS)', 'DE(ICRS)']].values

        print(self.stars_data.shape,' Stars loaded')

    def load_som_parameters(self, name = 'n67_90_tycho6'):

        #Load som from previusly trained model
        with open('../data/SOM_parameters/'+name+'/som1_'+ name + '.p', 'rb') as infile:
            try: self.som1, self.som2 = pickle.load(infile)
            except EOFError:
                print('Error: SOM not loaded')
                return
                

        #Load normalization parameters
        with open('../data/SOM_parameters/'+name+'/normalization_parameters_tycho' + str(name[-1]) + '.p', 'rb') as infile:
            try: self.norm_param = pickle.load(infile)
            except EOFError:
                print('Error: Normalization parameters not loaded')
                return

        #Load dictionary with the star features
        with open('../data/SOM_parameters/'+name+'/star_dict_'+ name + '.p', 'rb') as infile:
            try: self.star_dict_1, self.star_dict_2 = pickle.load(infile)
            except EOFError:
                print('Error: Star dictionary not loaded')
                return

        with open('../data/SOM_parameters/'+name+'/index_'+ name + '.p', 'rb') as infile:
            try: self.indices_dt = pickle.load(infile)
            except EOFError:
                print('Error: Index dictionary not loaded')
                return

        #Send a message if the load was successful
        print(type(self.som1))


    def compute_ids_predictions(self):
        self.indices_image = []
        self.predcited_stars = []

        for main_star in [x[0] for x in self.clusters_list]:

            #Order by distance to the main star at the loop
            self.stars_sorted_by_main, index_sort = order_by_main_dist(main_star, [x[0] for x in self.clusters_list], True)
            self.indices_image.append(index_sort[0:self.num_of_neirbours+1])
            
            stars_features_1, stars_features_2 = get_star_features(self.stars_sorted_by_main[0:self.num_of_neirbours+1],
                                                self.ref_pixel_to_deg, self.reference_FOV, self.recording_FOV)

            # Get prediction index
            predicted_star_ids_1 = predict_star_id(stars_features_1, self.norm_param[0:2], self.star_dict_1, self.som1)
            predicted_star_ids_2 = predict_star_id(stars_features_2, self.norm_param[2:4], self.star_dict_2, self.som2)
            
            # Get HIP of index
            hip_ids_predicted_1 = [self.stars_data[x][0].astype(int) for x in predicted_star_ids_1]
            hip_ids_predicted_2 = [self.stars_data[x][0].astype(int) for x in predicted_star_ids_2]

            # Get the intersection of the two predictions if there is only one star in common
            if len(list(set(predicted_star_ids_1).intersection(predicted_star_ids_2))) < 2 and len(
                list(set(predicted_star_ids_1).intersection(predicted_star_ids_2))) > 0:
                star_guess_index = list(set(predicted_star_ids_1).intersection(predicted_star_ids_2))[0]
                star_guess_id = list(set(hip_ids_predicted_1).intersection(hip_ids_predicted_2))[0]

            else:
                star_guess_index = None
                star_guess_id = None
            self.predcited_stars.append([star_guess_index, star_guess_id])

    def verify_predictions(self):

        indices_neigh_gt  = np.full((len(self.predcited_stars),self.num_of_neirbours+1 ), None)
        for i, predicted_star in enumerate(self.predcited_stars):
            if predicted_star[0] is not None:
                indices_neigh_gt[i]  = self.indices_dt[predicted_star[0]]

        indices_neigh_image = np.array([np.array(self.predcited_stars, dtype=object)[:,0][index] for index in self.indices_image], dtype=object)

        self.confirmed_stars_ids = check_star_id_by_neight( np.array(indices_neigh_gt, dtype=object),
                                                    np.array(indices_neigh_image, dtype=object),
                                                    np.array(self.indices_image, dtype=object),
                                                    True)

        self.confirmed_indices = [i for i in range(len(self.confirmed_stars_ids)) if self.confirmed_stars_ids[i] is not None]

    def compute_frame_position(self):
        img_center = np.array(self.frame.shape)/2
        distances = []
        for confirmed_index in self.confirmed_indices:
            dist_to_center = np.linalg.norm(img_center - self.clusters_list[confirmed_index]) * self.ref_pixel_to_deg * self.recording_FOV / self.reference_FOV
            distances.append(dist_to_center)
        self.frame_position = solve_point_c(self.stars_data[self.confirmed_stars_ids[self.confirmed_indices].tolist()][:,1:3], distances)
