import numpy as np

from minisom import MiniSom    

#  Create a set of 500 random stars in range [360,360]
n_stars = 500

# Create the set of points
x_points = np.random.uniform(0, 360, n_stars)
y_points = np.random.uniform(0, 360, n_stars)

# Combine the points
points = np.transpose([x_points, y_points])




data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      


som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100) # trains the SOM with 100 iterations
