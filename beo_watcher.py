from picamera import PiCamera
import cv2
import numpy as np
import datetime
import time
import shutil
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
import tokens
from _thread import *
import asyncio
import os
from images import *


current_dir     = os.path.dirname(__file__)
tmp_img_path    = os.path.join(current_dir,  "img.jpg")
tmp_video_base  = os.path.join(current_dir, "beo_video")
current_context:    ContextTypes.DEFAULT_TYPE
current_update:     Update


def log(mess: str):
	time_stamp = '{0:%Y-%m-%d %H:%M:%S} - '.format(datetime.datetime.now())
	print(time_stamp + mess)

# ======== TELEGRAM HANDLING =========
def init_telegram_bot():
	application = ApplicationBuilder().token(tokens.BOT_TOKEN).build()

	start_handler = CommandHandler('start', tg_cmd_start)
	application.add_handler(start_handler)

	current_handler = CommandHandler('current_state', tg_cmd_current_state)
	application.add_handler(current_handler)
	log("bot initialized")
	application.run_polling()


async def tg_cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
	global current_context
	global current_update
	log("start")
	current_context = context
	current_update  = update
	await context.bot.send_message(chat_id=update.effective_chat.id, text="Greatings new Béo's fan!")


async def tg_cmd_current_state(update: Update, context: ContextTypes.DEFAULT_TYPE):
	global current_context
	global current_update
	log("current_state")
	current_context = context
	current_update  = update
	try:
		img2 = os.path.join(current_dir, "img2.jpg")
		os.system("cp " + tmp_img_path + " " + img2) #so it is not rewritten while sending or something
		img = open(img2, "rb")
		await context.bot.send_photo(chat_id=update.effective_chat.id,  photo=img, write_timeout=100)
	except IOError:
		await context.bot.send_message(chat_id=update.effective_chat.id, text="No pictures taken yet")
	except Exception as e:
		await context.bot.send_message(chat_id=update.effective_chat.id, text="Unable to send the picture: " + str(e))


async def send_beo_to_followers(video_path: str):
	global current_context
	global current_update

	try:
		log("compressing video")
		video_out = os.path.join(current_dir, "out_vid.mp4")
		cmd = "rm " + video_out
		log(cmd)
		os.system(cmd)
		cmd = "ffmpeg -i " + video_path + " -vcodec libx265 -r 20 -preset ultrafast " + video_out
		log(cmd)
		os.system(cmd)
		log("sending:", video_out)
		video = open(video_out, "rb")
		await current_context.bot.send_video(chat_id=current_update.effective_chat.id,
											 video=video,
											 supports_streaming=True,
											 write_timeout=180,
											 caption="📸 the star is out 📸")
	except Exception as e:
		log(str(e))
		await current_context.bot.send_message(chat_id=current_update.effective_chat.id,
											   text="Béo detected, but unable to open video file")


def camera_loop():

	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)


	camera = PiCamera()
	camera.resolution 	= (1920, 1080)
	camera.vflip        = False
	camera.image_effect = 'none' #to reset it in case it was changed before


	while True:

		#sleep two seconds between every photo + should wait for the camera to be initialized
		time.sleep(2)

		camera.capture(tmp_img_path)

		img_to_test = cv2.imread(tmp_img_path)

		faces_detected = cat_faces_in_image(img_to_test)

		if len(faces_detected) > 0:

			log("béo detected")

			now = datetime.datetime.now()
			time_stamp = ' {0:%Y-%m-%d %H:%M:%S}'.format(now)

			# A cat has been detected, we start recording
			video_path = tmp_video_base + ".h264"
			camera.start_recording(video_path)
			camera.wait_recording(10)
			camera.stop_recording()

			log("finish recording")
			#TODO: do this in a separated thread
			# we copy it so we have the image that started the recording
			beo_detected = os.path.join(current_dir, "béo detected" + time_stamp)
			cv2.imwrite(beo_detected + ".jpg", img_to_test)

			##we add a cute hat over the cat's head because
			#(x,y,w,h) = faces_detected[np.argmax([w for (_,_, w,_) in faces_detected])]
			#hat_resized = cv2.resize(img_hat, (w, int(w * hat_ratio)))
			#overlay_image_alpha(img_to_test,
			#					hat_resized[:, :3],
			#					int(x - hat_resized.shape[1] * 0.05),
			#					int(y - hat_resized.shape[0] * 0.90),
			#					hat_resized[:, 3] / 255.0)
			#cv2.imwrite("béo with a hat detected" + time_stamp + ".jpg", img_to_test)


			asyncio.run(send_beo_to_followers(video_path))
			#we copy the video so we have a local backup
			os.system("cp '" + video_path + "' '" + beo_detected + ".mp4'")

		#img = cv2.imread(tmp_img_path)
		#cv2.imshow("gros tas", img)


if __name__ == "__main__":
	start_new_thread(camera_loop,())
	init_telegram_bot()



