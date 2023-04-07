import time
from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import telepot

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

webcam = cv2.VideoCapture(0)

token = '6029804352:AAGRqkd17D7lCXUi9i5GEZLqTnDcUbs7LlQ' # telegram token
receiver_id = 996219539 # https://api.telegram.org/bot<TOKEN>/getUpdates
bot = telepot.Bot(token)

if webcam.isOpened() == False:
	print('[!] error opening the webcam')

try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        if ret == True:
            detections = yolov7.detect(frame)
            detected_frame = draw(frame, detections)
            w = json.dumps(detections, indent=4)
            if json.dumps(detections, indent=4) in w == '[]':
                pass
            else:
                w = json.dumps(detections, indent=4).split('"')[3]
            # v = w.split('"')
            print(w)
            # print(json.dumps(detections, indent=4))
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
            if w == 'person':
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imwrite("screen.jpg", frame)
                bot.sendMessage(receiver_id, "Person Detected")
                filename = "screen.jpg"
                bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                time.sleep(1)
            elif w == 'knife':
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imwrite("screen2.jpg", frame)
                bot.sendMessage(receiver_id, "Danger: Knife Detected!!!")
                filename = "screen2.jpg"
                bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                time.sleep(1)
            else:
                pass
        else:
            break
except KeyboardInterrupt:
    pass

webcam.release()
print('[+] webcam closed')
yolov7.unload()