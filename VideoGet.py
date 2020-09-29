from threading import Thread
import cv2
import requests
import numpy as np
#import time
#import _thread as thread, queue, time
from PIL import Image, ImageOps





url = "http://192.168.86.27:8080/shot.jpg"


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        #self.stream = cv2.VideoCapture(src)
        self.stopped = False
        
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        self.img = cv2.imdecode(img_arr, -1)
        #(self.grabbed, self.frame) = self.stream.read()
        
        
    def start(self):    
        Thread(target=self.capturefromdevice, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):

        return  self.img
        

    def stop(self):
        self.stopped = True

       
    
    def capturefromdevice(self):
        while not self.stopped:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
            self.img = cv2.imdecode(img_arr, -1)
            
















