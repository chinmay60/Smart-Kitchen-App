import cv2
from VideoGet import VideoGet
from threading import Thread
from PIL import Image, ImageOps
import numpy as np  
#from tensorflow import keras
from home import *
from configparser import ConfigParser
from detect_cooker import *
import _thread as thread, queue, time
q = queue.Queue()
video_getter = VideoGet(src = "").start(q,)
frame = video_getter.read()
Thread(target=Detect_Cooker, args=(q,)).start()

	#img = cv2.imdecode(img, -1)
	#cv2.imshow("Image", img2)
	#cv2.waitKey(1)
	#print(img)
	
	#img_arr = np.array(img, dtype = np.uint8)
	#img = cv2.imdecode(img_arr, -1)
	#img = np.array(img)
	#cv2.imshow("Image", img)
	#cv2.waitKey(1)
#cv2.destroyAllWindows()
