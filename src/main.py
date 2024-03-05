# Get system path 
from calendar import c
import os

from scipy import cluster
path = os.getcwd()
# Get parent directory
parent = os.path.dirname(path)
#Add parent directory to system path
os.sys.path.insert(0, parent)

from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt
import yaml
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
    parser.add_argument('--train-path', dest='train_path', default="",
                        help='Path to SOM training data. Need to be specified')
    parser.add_argument('--catalog-folder', type=str, default="catalogs", help='catalog folder')
    parser.add_argument('--star-catalog-path', type=str, default="catalogs/Star_Catalog.csv", help='star catalog path')


    # Star detection filtering parameters
    parser.add_argument('--treshold-filter', type=float, default=0.0, help='treshold-filter value')
    parser.add_argument('--pixel-range', type=int, default=15, help='batch size')
    parser.add_argument('--mass-treshold', type=int, default=0.0, help='mass treshold')
    
    # Event frames parameters
    parser.add_argument('--accumulation-time-us', type=int, default=50000, help='accumulation time in us')
    parser.add_argument('--delta-t', type=int, default=10000, help='delta t in us')
    parser.add_argument('--buffer-size', type=int, default=10, help='buffer size')

    parser.add_argument('--load-params', dest='params_path', default="", 
        help="Path to the config.yaml file with the configuration parameters.")
    
    # Pixel to deg parameters 
    parser.add_argument('--pixel-to-deg', type=dict, help='pixel to degree parameters dictionary')

    # Verbosity and visualization
    parser.add_argument('--verbose', action='store_true',
                        help='if enabled, debug information will be printed')
    parser.add_argument('--show-video', action='store_true',
                        help='if enabled, video will be shown')
    parser.add_argument('--show-time', action='store_true',
                        help='if enabled, time will be shown')
    
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


def main():
    """ Main """

    start_time = time.time()    
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()  # Camera Geometry
            
    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=args.accumulation_time_us)

    # Init Cluster Frame and Video
    innit_fame = np.zeros((height, width), dtype=np.uint8)
    cluster_frame = ClusterFrame(innit_fame, pixel_to_deg = args.pixel_to_deg, pixel_range=args.pixel_range, treshold_filter=args.treshold_filter, mass_treshold=args.mass_treshold)
    cluster_frame.load_som_parameters(args.catalog_folder)
    cluster_frame.load_star_catalog(args.star_catalog_path)
    cluster_video = ClusterVideo()
    frames = []  # TODO: Change to np.array?

    def on_cd_frame_cb(ts, cd_frame):
        # window.show(cd_frame)
        frames.append(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    print('Initialisation time: ', time.time() - start_time)

    for evs in mv_iterator:

        event_frame_gen.process_events(evs)
        # TODO: Cambiar buffer a tiempo.

        if len(frames) == args.buffer_size:

            buffer_time = time.time()
            print('New max buffer: ', buffer_time - start_time)

            compact_frame = blend_buffer(frames, mirror=True)

            cluster_frame.update_clusters(compact_frame)

            cluster_frame.compute_ids_predictions()

            cluster_frame.verify_predictions()

            cluster_frame.compute_frame_position()

            print(cluster_frame.time_dict)
            cluster_frame.update_total_time()

            if args.verbose: 
                cluster_frame.info(show_time = args.show_time)
            if args.show_video:
                close_callbcak = cluster_video.update_frame(cluster_frame.plot_cluster_cv(show_confirmed_ids=True))
                if close_callbcak:
                    cv2.destroyAllWindows()
                    break

            cluster_frame.time_dict 
                    
            frames.clear()

    cluster_frame.print_total_time()

if __name__ == "__main__":
    main()