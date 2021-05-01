import sys
sys.path.insert(0,'../libs/OpencvToolsKit')
from tools import *
import numpy as np
import cv2

#0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
#1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
#3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#5. CV_CAP_PROP_FPS Frame rate.
#6. CV_CAP_PROP_FOURCC 4-character code of codec.
#7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
#14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
#18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

def test_camera(color=True):
    cap = cv2.VideoCapture(0)
    print("CV_CAP_PROP_FORMAT",cap.get(cv2.CAP_PROP_FORMAT))
    print("CV_CAP_PROP_MODE",cap.get(cv2.CAP_PROP_MODE))
    print("CV_CAP_PROP_FPS",cap.get(cv2.CAP_PROP_FPS))
    print("CV_CAP_PROP_CONTRAST",cap.get(cv2.CAP_PROP_CONTRAST))
    print("CV_CAP_PROP_GAIN",cap.get(cv2.CAP_PROP_GAIN))
    print("CV_CAP_PROP_FRAME_WIDTH",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("CV_CAP_PROP_FRAME_HEIGHT",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("CV_CAP_PROP_POS_FRAMES",cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("CV_CAP_PROP_EXPOSURE",cap.get(cv2.CAP_PROP_EXPOSURE))
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()


        # Our operations on the frame come here
        if color is False:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


test_camera(color=True)
