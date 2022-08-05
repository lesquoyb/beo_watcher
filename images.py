import cv2
import numpy as np
import datetime
import time
import shutil
import os

current_dir     = os.path.dirname(__file__)
face_cascade    = cv2.CascadeClassifier(os.path.join(current_dir,'resources/cat.xml'))
img_hat         = cv2.imread(os.path.join(current_dir, "resources/hat.png"))
hat_ratio       = img_hat.shape[0] / img_hat.shape[1]


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


def cat_faces_in_image(img_to_test):

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img_to_test, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.3,
                                          minNeighbors=10,
                                          minSize=(300, 300))
    return faces

if __name__ == "__main__":
    
    img_to_test = cv2.imread("test_detect.jpg")
    faces_detected = cat_faces_in_image(img_to_test)
    
    #we add a cute hat over the cat's head because
    (x,y,w,h) = faces_detected[np.argmax([w for (_,_, w,_) in faces_detected])]
    print(x,y, w,h, hat_ratio)
    hat_resized = cv2.resize(img_hat, (w, int(w * hat_ratio)))
    print(hat_resized.shape)
    overlay_image_alpha(img_to_test,
                        hat_resized[:, :3],
                        int(x - hat_resized.shape[1]),
                        int(y - hat_resized.shape[0]),
                        hat_resized[:, 3] / 255.0)
    cv2.imwrite("test béo with a hat detected"  + ".jpg", img_to_test)

    
    
