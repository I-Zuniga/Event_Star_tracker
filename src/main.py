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
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry
            
    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=50000)


    frames = []

    def on_cd_frame_cb(ts, cd_frame):
        # window.show(cd_frame)
        frames.append(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    cont = 0
    threshold = 0.7  # Percentage of the maximum value to consider a pixel as a star
    pixel_range = 20 # Number of pixels to consider around the maximum value

    for evs in mv_iterator:
        event_frame_gen.process_events(evs)

        if len(frames) == 10:
            compact_frame = blend_buffer(frames)
            
            frame_thr = cv2.threshold(compact_frame, np.max(compact_frame)*threshold, 1, cv2.THRESH_TOZERO)[1]

            clusters =  max_value_cluster(frame_thr, pixel_range, 20)
            clusters = sorted(clusters, key=lambda x: x[1], reverse=True)
            clusters_index = np.array([cluster[0] for cluster in clusters])
            clusters_index = sorted(clusters_index, key=lambda x: x[1], reverse=True)
            clusters = index_cluster(compact_frame, pixel_range, clusters_index) 

            # print(clusters)
            cont += 1
            print('Computing clusters')

            frames = []
    
        # print(frames)

if __name__ == "__main__":
    main()