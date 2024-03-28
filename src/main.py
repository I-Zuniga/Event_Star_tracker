# Get system path 
import os
path = os.getcwd() # Get parent directory
parent = os.path.dirname(path)#Add parent directory to system path
os.sys.path.insert(0, parent)

import numpy as np
import yaml

import multiprocessing
import time

from lib.utils import *
from lib.plot_utils import *   
from lib.event_processing import *
from lib.real_time_video import *

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent

import cv2

def parse_args():
    import argparse
  
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # Paths 
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument('--train-folder', type=str, default="catalogs", help='catalog folder')
    parser.add_argument('--star-catalog-path', type=str, default="catalogs/Star_Catalog.csv", help='star catalog path')
    parser.add_argument('-l', '--load-params', dest='params_path', default="", 
        help="Path to the config.yaml file with the configuration parameters.")


    # Star detection filtering parameters
    parser.add_argument('--treshold-filter', type=float, default=0.0, help='treshold-filter value')
    parser.add_argument('--pixel-range', type=int, default=15, help='batch size')
    parser.add_argument('--mass-treshold', type=int, default=0.0, help='mass treshold')
    parser.add_argument('--max-n-clusters', type=int, default=30, help='max number of clusters')
    parser.add_argument('--compute-ids', action='store_true', default=True, help='if enabled, ids will be computed')
    
    # Event frames parameters
    parser.add_argument('--accumulation-time-us', type=int, default=50000, help='accumulation time in us')
    parser.add_argument('--delta-t', type=int, default=10000, help='delta t in us')
    parser.add_argument('--buffer-size', type=int, default=10, help='buffer size')
    
    # Pixel to deg parameters 
    parser.add_argument('--pixel-to-deg', type=dict, help='pixel to degree parameters dictionary')

    # Verbosity and visualization
    parser.add_argument('--verbose', action='store_true',
                        help='if enabled, debug information will be printed')
    parser.add_argument('--show-video', action='store_true',
                        help='if enabled, video will be shown')
    parser.add_argument('--show-time', action='store_true',
                        help='if enabled, time will be shown')
    
    # Data saving
    parser.add_argument('--save-stars', action='store_true',
                        help='if enabled, stars will be saved')
    parser.add_argument('--save-attitude', action='store_true',
                        help='if enabled, attitude (RA, DE) of the image will be saved')
    
    args = parser.parse_args()

    if args.params_path:
        print(f"Loading parameters from {args.params_path}")
        with open(args.params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
        for arg_name, arg_params in params['parameters'].items():
            if arg_name in args:
                setattr(args, arg_name, arg_params)
    else:
        print(f"No parameters file found at {args.params_path}. Using default values, erors may occur.")

    return args

def data_timer( args, send_time, terminate_event):
    '''Send data every "time" seconds.'''
    start_time = time.time()
    loop_time = time.time()

    while not terminate_event.is_set():
        current_time= time.time()
        if current_time - loop_time >= send_time:
            if not data_queue.empty():
                if args.save_stars:
                    # TODO: save_stars_queue = queue.Queue()
                    # ClusterFrame.save_stars(
                    #     save_stars_queue.get(),
                    #     training_name = args.train_folder, 
                    #     recording_path=args.input_path, 
                    #     time = time.time() - start_time)
                    print('save_stars Not implemented yet.')
                    
                if args.save_attitude:
                    ClusterFrame.save_attitude(
                        data_queue.get(),
                        training_name = args.train_folder, 
                        recording_path=args.input_path, 
                        time = time.time() - start_time
                        )
                    print('saving position', data_queue.get())
            loop_time = time.time() # Reset the timer

def run_star_tracker(args):
    """ Main """

    start_time = time.time()    

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()  # Camera Geometry
            
    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=args.accumulation_time_us)

    # Init Cluster Frame and Video
    innit_fame = np.zeros((1,height, width), dtype=np.uint8)
    cluster_frame = ClusterFrame(
        innit_fame,
        pixel_to_deg = args.pixel_to_deg,
        pixel_range=args.pixel_range, 
        treshold_filter=args.treshold_filter, 
        mass_treshold=args.mass_treshold, 
        max_number_of_clusters=args.max_n_clusters)
    
    cluster_frame.load_som_parameters(args.train_folder)
    cluster_frame.load_star_catalog(args.star_catalog_path)
    cluster_video = ClusterVideo()
    global frames
    frames = np.empty((0, height, width), dtype=np.uint8) 
    # frames = np.random.randint(0, 256, size=(1, height, width), dtype=np.uint8)
    # Create a random frame to initialize the frames variable with np.random.randint

    

    def on_cd_frame_cb(ts, cd_frame):
        global frames  # Add this line to access the global frames variable
        gray_frame = np.reshape(cv2.cvtColor(cd_frame, cv2.COLOR_BGR2GRAY), (1, height, width))
        frames = np.append(frames,gray_frame, axis=0)  # Append the gray_frame as a new row to the frames array
        
    event_frame_gen.set_output_callback(on_cd_frame_cb)

    cluster_frame.init_compilation()

    print('Initialisation time: ', time.time() - start_time)

    for evs in mv_iterator:

        event_frame_gen.process_events(evs)
        # TODO: Cambiar buffer a tiempo.
        # Options:
        # best effort 
        # garantize data each second 
        # resaerch benchamarck in LIS and recursive

        # PowerConsumption 

        if len(frames) == args.buffer_size:

            buffer_time = time.time()

            #--------------------#
            # Compuatation calls #
            #--------------------#

            cluster_frame.update_clusters(frames)

            if args.compute_ids:

                cluster_frame.compute_ids_predictions_2()

                cluster_frame.verify_predictions()

                cluster_frame.compute_frame_position()

                data_queue.put(cluster_frame.frame_position) # Send data to the queue

                cluster_frame.update_total_time() # Can be commented to improve performance

            
            # DuckIp o noDNS 
            #-----------------------------------#
            # Visualization, verbose and saving #
            #-----------------------------------#
            if args.verbose: 
                print('-'*50)
                print('New max buffer: ', buffer_time - start_time)
                # cluster_frame.info(show_time = args.show_time)

            if args.show_video:
                close_callbcak = cluster_video.update_frame(cluster_frame.plot_cluster_cv(show_confirmed_ids=True))
                if close_callbcak:
                    cv2.destroyAllWindows()
                    break
            
            frames = np.empty((0, height, width), dtype=np.uint8) 
            # frames.clear()

    cluster_frame.print_total_time()


data_queue = multiprocessing.Queue() # Attitude data queue

def main():
    args = parse_args()

    # Event to signal other processes to terminate
    terminate_event = multiprocessing.Event()
    
    star_tracker_process = multiprocessing.Process(target=run_star_tracker, args=(args,))
    # data_process = multiprocessing.Process(target=data_timer, args=(args, 1.0, terminate_event))

    star_tracker_process.start()
    # data_process.start()

    # Wait for the star_tracker_process to terminate
    star_tracker_process.join()

    # Set the terminate event to signal other processes to terminate
    terminate_event.set()

    # Wait for the data_process to terminate
    # data_process.join()


if __name__ == "__main__":
    main()