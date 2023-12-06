#Utils from event camera attitude determination 
import pandas as pd
import numpy as np 

# set diferent type of features combinations

class features: 
    def __init__(self, type):
        self.type = type 
        pass




def load_catalog(file_path):
    """
    Reads a Star catalog CSV file with a header.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Return the DataFrame
        return df

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file at path {file_path} is empty.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the CSV file at path {file_path}. Please check the file format.")
    except ValueError as e:
        print(e)

def random_dataset(n_stars = 4000):
    
    """
    Create a set of random stars in range [360,180].

    Parameters:
    - dim (int): Number of points in the dataset.

    Returns:
    - stars_dataset: A array containing a ramdom dataset of stars [id, RA [deg], DE [deg]].
    """
    # Create the set of points
    x = np.random.uniform(0, 360, n_stars)
    y = np.random.uniform(0, 180, n_stars)
    stars_id = np.linspace(0,n_stars-1,n_stars, dtype=int)

    stars_dataset = np.array( np.transpose([stars_id,x,y]))

    return stars_dataset

def get_star_dataset(type='random', path=None, n_stars=None):

    """
    Return a star catalog of selected type

    Parameters:
    - type (str): Catalog type:
        'random': Random dataset
        'tycho': Brightest stars in tycho 2018 dataset: dim = 9925
    - path (str): Path to the CSV file
    - n_stars (int): Random number of stars

    Returns:
    - dataset (see further functions).
    """
    try:
        if type == 'random':
            return random_dataset(n_stars=n_stars)
        elif type == 'tycho':
            return load_catalog(path)
        else:
            raise ValueError('Invalid type')
    except Exception as e:
        print(f'Error: {e}')