from picamera import PiCamera
import time
import cv2
import numpy as np

tmp_img_path    = "img.jpg"
tmp_video_base  = "beo_video"
face_cascade    = cv2.CascadeClassifier('resources/cat.xml')


# Source: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
def overlay_image_alpha(img, img_overlay, x_off, y_off, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y_off), min(img.shape[0], y_off + img_overlay.shape[0])
    x1, x2 = max(0, x_off), min(img.shape[1], x_off + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y_off), min(img_overlay.shape[0], img.shape[0] - y_off)
    x1o, x2o = max(0, -x_off), min(img_overlay.shape[1], img.shape[1] - x_off)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def cat_faces_in_image(image_path: str):
    # reads frames from a image
    img_to_test = cv2.imread(image_path)

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img_to_test, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
    return faces


def beo_detected(image_path: str):
    return len(cat_faces_in_image(image_path)) > 1

def send_beo_to_followers():
    #TODO
    pass

if __name__ == "__main__":

    camera = PiCamera()
    camera.resolution = (1920, 1080)
    camera.vflip        = True
    camera.image_effect = 'none' #to reset it in case it was changed before


    while True:

        #sleep two seconds between every photo + should wait for the camera to be initialized
        time.sleep(2)

        camera.capture(tmp_img_path)

        if beo_detected(tmp_img_path):
            print("béo detected")
            video_path = tmp_video_base + str(time.time()) + ".h264"
            camera.start_recording(video_path)
            camera.wait_recording(5)
            camera.stop_recording()

            send_beo_to_followers()

        #img = cv2.imread(tmp_img_path)
        #cv2.imshow("gros tas", img)

