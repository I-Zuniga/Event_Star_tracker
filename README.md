# Event_Star_tracker
 Star traking Attitude Determination via event camara and Self Orginized Networks 

# Requirements 

Some parts of this repository use metavision sdk libary, please referer to the offical installation instructions: https://docs.prophesee.ai/stable/index.html

The rest of the dependecies are in: requirements.txt

# Structure

## Repository Structure

- **data**
    - **catalogs:** Contains CSV files with data about different catalogs.
        - `tycho2_VT_4.csv`
        - `tycho2_VT_5.csv`
        - `tycho2_VT_6.5.csv`
        - `tycho2_VT_6.csv`
        - `tycho2_VT_7.csv`
    - **recordings:** Houses several subdirectories for recordings from different dates, containing raw recordings and associated index files. All the recording can not be uploaded for size issues. One example of a star field and one example of a sun calibration recording is provided.
        - **02_02** contains recordings and sun calibration data from February 2, 2024.

    - **SOM_parameters:** Contains Self-Organizing Map (SOM) parameters and data files related to various experiments.
        - Multiple subdirectories each containing index files, normalization parameters, and other data.

- **jupyter_notebooks**
    - Contains Jupyter Notebooks related to various analyses and tests for the project, including performance analysis, runtime comparison, and training.
        - `features_analisys.ipynb`
        - `magnitude_star_filtering_test.ipynb`
        - `performace_plots.ipynb`
        - `rotation_speed_vs_detection.ipynb`
        - `runtime_comparison.ipynb`
        - `sdk_test.ipynb`: Test of the event_processing functions 
        - `simple_nn.ipynb`
        - `stars_identified_plot.png`
        - `sun_calibration.ipynb`
        - `test_multi_som.ipynb`: Perfromance test of the SOM
        - `train_multi_som.ipynb`: Main train of the SOMs

- **lib**
    - Contains library files that support the project's codebase, including utility and plotting functions.
        - `event_processing.py`: General functions for event proccesing and star identification
        - `plot_utils.py`: Utils for plotting using matplolib and opencv 
        - `real_time_video.py`: Library with the main classes
        - `som_training.py`: Utils for the training of the SOM with Optuna 
        - `utils.py`: General utils 

- **requirements**
    - Contains a requirements file with project dependencies.
        - `requirements.txt`

- **sdk_tutorials**
    - Contains Python scripts related to SDK tutorials for getting started with the project's SDK and check the installation.
        - `detection_and_tracking_pipeline.py`
        - `metavision_sdk_get_started.py`
        - `metavision_sdk_noise.py`

- **src**
    - Contains source code files for the project.
        - `main.py`
        - `parameters.yaml`
        - `sun_calibration.py`

- **stellarium_script**
    - Contains scripts for the Stellarium software.
        - `constant_rotation.ssc`
        - `constant_simple.ssc`

- **training_optuna**
    - Contains Optuna optimization-related data and scripts, including a SQLite database with the diferent optimization perfromed.
        - `db.sqlite3`
        - `optuna_som_opt.py`


# To run 

Go to the src folder

Adding the arguments directly from the command line: 

    python3 main.py -i ../data/recordings/02_02/recording_2024-02-02_17-21-24_FOV14.3_MAG6.raw --show-video --other-args

Via the parameters.yaml:

    python3 main.py --load-params ./parameters.yaml


# Optuna
To run the optuna dasboard:

    optuna-dashboard sqlite:///db.sqlite3 

