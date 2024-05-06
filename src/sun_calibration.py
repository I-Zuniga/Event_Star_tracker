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
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=10000)

    frames = []

    def on_cd_frame_cb(ts, cd_frame):
        frames.append(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    sun_aparent_diameter = 0.54152 # From stelarium [degrees]

    max_buffer = 1000

    for evs in mv_iterator:
        event_frame_gen.process_events(evs)

        if len(frames) == max_buffer:

            frame = calibration_blend(frames).astype(np.uint8)

            cv2.imshow("Original frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #################### filters set one 
            frame = cv2.threshold(frame*255, 120, 1, cv2.THRESH_BINARY)[1]
            frame = frame*255
            for i in range(1):
                frame = cv2.blur(frame,(5,5))

            frame = cv2.threshold(frame*255, 100, 1, cv2.THRESH_BINARY)[1]

            frame = frame*255
            for i in range(2):
                frame = cv2.blur(frame,(5,5))

            ################### filters set two
            # frame = cv2.threshold(frame, 120, 1, cv2.THRESH_TOZERO)[1]

            # for i in range(5):
            #     frame = cv2.blur(frame,(5,5))
            #     frame = cv2.threshold(frame, 5, 1, cv2.THRESH_TOZERO)[1]

            # frame = cv2.threshold(frame*255, 120, 1, cv2.THRESH_BINARY)[1]
            # frame = frame*255

            # for i in range(10):
            #     frame = cv2.blur(frame,(5,5))
            
            image = frame
            # plot_image(image)

            output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gray =  image

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000,
						   param1=15,param2=15,minRadius=100,maxRadius=400)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for (x, y, r) in circles:

                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                    print(f"Circle center: {x,y} , radius: {r}")


                sun_aparent_diameter = 0.54152 # From stelarium [degrees]
                print('Circle detected, diameter: ', r*2, ' [pixels], pixel to degree ratio: '
                        , sun_aparent_diameter/(r*2), ' [degrees/pixel]', sun_aparent_diameter/(r*2)*3600, ' [arcsec/pixel]')

                # show the output image
                cv2.imshow("output", output)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            else:
                print("No circles found")


            frames = []
    
        # print(frames)

if __name__ == "__main__":
    main()