import cv2
import numpy as np

"""
    Returns largest contour coordinates, and contour type (cone or cube).
    Cone: 1
    Cube: 0
"""
def get_largest_contour(cone_contours, cube_contours):
    if len(cone_contours) == 0:
        return (max(cube_contours, key=cv2.contourArea), 0)
    elif len(cube_contours) == 0:
        return (max(cone_contours, key=cv2.contourArea), 1)
    else:
        largest_cone_contour = max(cone_contours, key=cv2.contourArea)
        largest_cube_contour = max(cube_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cone_contour) > cv2.contourArea(largest_cube_contour):
            return (largest_cone_contour, 1)
        else:
            return (largest_cube_contour, 0)


# runPipeline() is called every frame by Limelight's backend.
"""
    Returns largest contour for the LL crosshair, the modified image, and custom robot data (whether the contour is a cone or cube).
    Cone: 1
    Cube: 0
"""
def runPipeline(image, llrobot):
    # convert the input image to the HSV color space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # CONSTANTS THAT MUST BE TUNED
    # convert the hsv to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    CONE_THRESHOLD = cv2.inRange(img_hsv, (43,228,61), (62, 255, 255))
    CUBE_THRESHOLD = cv2.inRange(img_hsv, (82,240,42), (92, 255, 64))
    SPECKLE_THRESHOLD = 820

    # find contours in the new binary image
    cone_contours, _ = cv2.findContours(CONE_THRESHOLD, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cube_contours, _ = cv2.findContours(CUBE_THRESHOLD, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = np.array([[]])
    largest_contour_type = None

    # values to send back to the robot
    llpython = []

    if len(cone_contours) > 0 or len(cube_contours) > 0:
        largest_contour, largest_contour_type = get_largest_contour(cone_contours, cube_contours)
        llpython = [largest_contour_type]
        x,y,w,h = cv2.boundingRect(largest_contour)
        if h * w < SPECKLE_THRESHOLD:
            # detected speckles, ignore the contours
            largest_contour = np.array([[]])
            llpython = []
            cv2.putText(image, 'No contours', (0, 230), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # draw the unrotated bounding box
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            # draw the raw contours
            # cv2.drawContours(image, cone_contours, -1, 255, 2)
            # cv2.drawContours(image, cube_contours, -1, 255, 2)
            # put type of contour detected as text
            if largest_contour_type == 0:
	            cv2.putText(image, 'Cone detected', (0, 230), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
            elif largest_contour_type == 1:
                cv2.putText(image, 'Cube detected', (0, 230), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
       cv2.putText(image, 'No contours', (0, 230), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
    return largest_contour, image, llpython
