from picamera import PiCamera
import time
import cv2

camera = PiCamera()
camera.resolution = (1920, 1080)
camera.vflip = True
camera.image_effect = 'none'
tmp_img_path = "img.jpg"
tmp_video_base = "beo_video"

def beo_detected():
    return True #TODO

while True:
    
    #sleep two seconds between every photo + should wait for the camera to be initialized
    time.sleep(2)

    camera.capture(tmp_img_path)
    
    if beo_detected():
        video_path = tmp_video_base + str(time.time()) + ".h264"
        camera.start_recording(video_path)
        camera.wait_recording(5)
        camera.stop_recording()
        
        
    #img = cv2.imread(tmp_img_path)
    #cv2.imshow("gros tas", img)

