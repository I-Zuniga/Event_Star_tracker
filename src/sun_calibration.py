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
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, 
                                                       accumulation_time_us=10000)


    frames = []

    def on_cd_frame_cb(ts, cd_frame):
        # window.show(cd_frame)
        frames.append(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    cont = 0

    sun_aparent_diameter = 0.54152 # From stelarium [degrees]
    FOV = 2                # From stelarium (Depends on zoom) [degrees]


    max_buffer = 700

    for evs in mv_iterator:
        event_frame_gen.process_events(evs)

        if len(frames) == max_buffer:
            compact_frame = calibration_blend(frames).astype(np.uint8)
            
            frame = cv2.threshold(compact_frame, 100, 1, cv2.THRESH_TOZERO)[1]

            for i in range(100):
                frame = cv2.blur(frame,(5,5))
                frame = cv2.threshold(frame, 5, 1, cv2.THRESH_TOZERO)[1]
            
            circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 300,
						   param1=10,param2=30,minRadius=200,maxRadius=600)

            if circles is not None:

                if len(circles) > 1:
                    print("Error: More than one circle detected")
                else:
                    print('Circle detected, diameter: ', circles[0][0][2]*2, ' [pixels], pixel to degree ratio: '
                        , sun_aparent_diameter/circles[0][0][2]*2, ' [degrees/pixel]',    'FOV: ', FOV, ' [degrees]')
            else:
                print('No circle detected')

            frames = []
    
        # print(frames)

if __name__ == "__main__":
    main()