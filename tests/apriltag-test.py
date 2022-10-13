import cv2
import numpy as np
import apriltag

def main():
    input_steam = cv2.VideoCapture(0)

    ret, input_img = input_steam.read()
    output_img = np.copy(input_img)

    # apriltag detector
    detector = apriltag("tag16h5")

    height, width, channels = input_img.shape

    x_mid = width // 2
    y_mid = height // 2

    while True:

        ret, input_img = input_steam.read()
        output_img = np.copy(input_img)

        detections = detector.detect(input_img)

        print(detections)

        # cv2.circle(output_img, center=(x_mid, y_mid), radius=3, color=(0, 0, 255), thickness=-1)
        #
        # print(x_mid, " ", y_mid)

        cv2.imshow("frame", output_img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    input_steam.release()
    cv2.destroyAllWindows()
