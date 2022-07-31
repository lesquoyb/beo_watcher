from picamera import PiCamera
import cv2
import numpy as np
import datetime
import time
import shutil
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
import tokens


tmp_img_path    = "img.jpg"
tmp_video_base  = "beo_video"
face_cascade    = cv2.CascadeClassifier('resources/cat.xml')
img_hat         = cv2.imread("resources/hat.png")
hat_ratio       = img_hat.shape[0] / img_hat.shape[1]
current_context:    ContextTypes.DEFAULT_TYPE
current_update:     Update


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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
    return faces


# ======== TELEGRAM HANDLING =========
def init_telegram_bot():
    application = ApplicationBuilder().token(tokens.BOT_TOKEN).build()

    start_handler = CommandHandler('start', tg_cmd_start)
    application.add_handler(start_handler)

    current_handler = CommandHandler('current_state', tg_cmd_current_state)
    application.add_handler(current_handler)

    application.run_polling()


async def tg_cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_context
    global current_update
    current_context = context
    current_update  = update
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Béo connected!")


async def tg_cmd_current_state(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_context
    global current_update
    current_context = context
    current_update  = update
    await context.bot.send_message(chat_id=update.effective_chat.id, text="alles gut")
    await context.bot.send_photo(chat_id=update.effective_chat.id,  photo=tmp_img_path)
    print("current_state")

def send_beo_to_followers(video_path: str):
    global current_context
    global current_update
    current_context.bot.send_video(chat_id=current_update.effective_chat.id, video=video_path)

if __name__ == "__main__":

    init_telegram_bot()

    camera = PiCamera()
    camera.resolution = (1024, 720)
    camera.vflip        = True
    camera.image_effect = 'none' #to reset it in case it was changed before


    while True:

        #sleep two seconds between every photo + should wait for the camera to be initialized
        time.sleep(2)

        camera.capture(tmp_img_path)

        img_to_test = cv2.imread(tmp_img_path)

        faces_detected = cat_faces_in_image(img_to_test)

        if len(faces_detected) > 0:

            print("béo detected")

            now = datetime.datetime.now()

            # A cat has been detected, we start recording
            video_path = tmp_video_base + ' {0:%Y-%m-%d %H:%M:%S}'.format(now) + ".h264"
            camera.start_recording(video_path)
            camera.wait_recording(5)
            camera.stop_recording()

            #TODO: do this in a separated thread
            # we copy it so we have the image that started the recording
            cv2.imwrite("béo detected" + ' {0:%Y-%m-%d %H:%M:%S}'.format(now) + ".jpg", img_to_test)

            #we add a cute hat over the cat's head because
            (x,y,w,h) = faces_detected[np.argmax([w for (_,_, w,_) in faces_detected])]
            hat_resized = cv2.resize(img_hat, (w, int(w * hat_ratio)))
            overlay_image_alpha(img_to_test, hat_resized[:, :, :3], int(x - hat_resized.shape[1] * 0.05), int(y - hat_resized.shape[0] * 0.90), hat_resized[:, :, 3] / 255.0)
            cv2.imwrite("béo with a hat detected" + ' {0:%Y-%m-%d %H:%M:%S}'.format(now) + ".jpg", img_to_test)

            send_beo_to_followers(video_path)

        #img = cv2.imread(tmp_img_path)
        #cv2.imshow("gros tas", img)

