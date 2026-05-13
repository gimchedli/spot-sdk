import cv2
from pupil_apriltags import Detector
import numpy as np

def find_marker():
    # Get the video stream
    cap = cv2.VideoCapture(0)

    # Set up apriltag detector
    detector = Detector(families="tag36h11")

    # Some drawing params
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1
    txt_c = (255, 0, 0)

    rect_c = (0, 0, 255)
    rect_t = 2

    while True:
        # Get the image and convert to grayscale for marker detection
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Find any tags in the image
        tags = detector.detect(gray)

        # If a tag is found
        if tags:
            # Extract tag info
            tag = tags[0]
            corners = np.array(tag.corners)
            tag_id = tag.tag_id

            # Create a rectangle around the marker and add some text
            rect = cv2.minAreaRect(np.float32(corners))
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            top_left_corner = corners[3]
            x, y = int(top_left_corner[0]), int(top_left_corner[1])

            text = f"Maker ID: {tag_id}"
            cv2.putText(img=frame, text=text, org=(x, y-10), fontFace=font, fontScale=font_scale,
                        color=txt_c, thickness=font_thick)
            cv2.drawContours(frame, [box], 0, rect_c, rect_t)


        cv2.imshow('Camera', frame)
        
        # Close the video stream if the 'q' is pressed 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()


def main():
    find_marker()

if __name__ == "__main__":
    main()