import os
path = os.getcwd()
# Get parent directory
parent = os.path.dirname(path)
#Add parent directory to system path
os.sys.path.insert(0, parent)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.utils import *
from lib.plot_utils import *   
from lib.event_processing import *
from lib.real_time_video import *

def plot_sphere_with_trajectory(radius=1, center=(0, 0, 0), num_samples=100, trajectory=None, star_catalog=None):
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
    ax.plot_surface(x, y, z, color='b', alpha=0.3)

    # Set aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Plot the trajectory if provided
    if trajectory is not None:
        latitudes, longitudes = trajectory
        # Convert latitude and longitude to radians
        latitudes_rad = np.radians(latitudes)
        longitudes_rad = np.radians(longitudes)
        
        # Convert latitude and longitude to Cartesian coordinates on the sphere. 1.01 is used to make the trajectory visible
        x_traj = radius*1.01 * np.cos(latitudes_rad) * np.cos(longitudes_rad) + center[0]
        y_traj = radius*1.01 * np.cos(latitudes_rad) * np.sin(longitudes_rad) + center[1]
        z_traj = radius*1.01 * np.sin(latitudes_rad) + center[2]
        
        # Plot the trajectory
        ax.plot(x_traj, y_traj, z_traj, color='r', linestyle='-')

    # Plot the x, y, and z axes
    ax.plot([center[0], center[0] + radius*1.1], [center[1], center[1]], [center[2], center[2]], color='g', linestyle='-', linewidth=1)
    ax.plot([center[0], center[0]], [center[1], center[1] + radius*1.1], [center[2], center[2]], color='g', linestyle='-', linewidth=1)
    ax.plot([center[0], center[0]], [center[1], center[1]], [center[2], center[2] + radius*1.1], color='g', linestyle='-', linewidth=1)
    # plot the axis name in the end of the axis
    ax.text(center[0] + radius*1.1, center[1], center[2], 'X', color = 'g')
    ax.text(center[0], center[1] + radius*1.1, center[2], 'Y', color = 'g')
    ax.text(center[0], center[1], center[2] + radius*1.1, 'Z', color = 'g')
    

    # Plot the star catalog if provided
    if star_catalog is not None:
        # Convert latitude and longitude to radians
        latitudes_rad = np.radians(star_catalog[:, 0])
        longitudes_rad = np.radians(star_catalog[:, 1])
        
        # Convert latitude and longitude to Cartesian coordinates on the sphere
        x_stars = radius * np.cos(latitudes_rad) * np.cos(longitudes_rad) + center[0]
        y_stars = radius * np.cos(latitudes_rad) * np.sin(longitudes_rad) + center[1]
        z_stars = radius * np.sin(latitudes_rad) + center[2]
        
        # Plot the stars
        ax.scatter(x_stars, y_stars, z_stars, color='y', s=5)


    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #Hide axis scale
    # ax.set_axis_off()


    # Show plot
    plt.show()

# Example usage:
# Generate some random latitude and longitude coordinates for the trajectory
num_epochs = 100
latitudes = np.linspace(0, 10, num_epochs)
longitudes = np.linspace(0, 10, num_epochs)


catalog_path = '../data/catalogs/tycho2_VT_6.csv'
stars_data = get_star_dataset(type ='tycho', path = catalog_path)

# If data type is a dataframe from a catalog transform it to array
if isinstance(stars_data, pd.DataFrame):
    # stars_data['data_number'] = stars_data.index
    stars_data = stars_data[['RA(ICRS)', 'DE(ICRS)']].values

# print("Data shape: ", stars_data.shape)

trajectory = (latitudes, longitudes)

# Plot sphere with trajectory
plot_sphere_with_trajectory(radius=1, trajectory=trajectory, star_catalog=stars_data)
