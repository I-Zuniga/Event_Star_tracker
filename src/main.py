# Get system path 
import os
path = os.getcwd()
# Get parent directory
parent = os.path.dirname(path)
#Add parent directory to system path
os.sys.path.insert(0, parent)

from metavision_core.event_io import EventsIterator
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument('--train-path', dest='train_path', default="",
                        help='Path to SOM training data. Need to be specified')
    parser.add_argument('--treshold-filter', type=float, default=0.0, help='treshold-filter value')
    parser.add_argument('--pixel-range', type=int, default=15, help='batch size')
    parser.add_argument('--show-video', action='store_true',
                        help='if enabled, video will be shown')
    args = parser.parse_args()
    # TODO: Config file for parameters
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry
            
    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=50000)

    # Init Cluster Frame and Video
    innit_fame = np.zeros((height, width), dtype=np.uint8)
    cluster_frame = ClusterFrame(innit_fame, pixel_range=15, treshold_filter=0.3, mass_treshold=0.2)
    cluster_frame.load_som_parameters()
    cluster_frame.load_star_catalog()
    cluster_video = ClusterVideo()
    frames = []  # TODO: Change to np.array?

    def on_cd_frame_cb(ts, cd_frame):
        # window.show(cd_frame)
        frames.append(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    for evs in mv_iterator:
        event_frame_gen.process_events(evs)

        # TODO: Cambair buffer a tiempo.

        if len(frames) == 10:
            compact_frame = blend_buffer(frames, mirror=True)            
            cluster_frame.update_clusters(compact_frame)
            cluster_frame.compute_ids_predictions()
            cluster_frame.verify_predictions()
            print(cluster_frame.confirmed_stars_ids)
            if args.show_video:
                close_callbcak = cluster_video.update_frame(cluster_frame.plot_cluster_cv(show_con_ids=True))
                if close_callbcak:
                    cv2.destroyAllWindows()
                    break
                    
            frames.clear()


if __name__ == "__main__":
    main()