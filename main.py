
import cv2
from VideoGet import VideoGet
from threading import Thread
from PIL import Image, ImageOps
import numpy as np  
from tensorflow import keras
from home import *
from configparser import ConfigParser





google_model = keras.models.load_model('model/newkeras_model.h5', compile = False)

    
#def putIterationsPerSec(frame, iterations_per_sec):
"""
    Add iterations per second text to lower-left corner of a frame.
"""

    #cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        #(10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    #return frame



def threadVideoGet(source):
    #x,y,w,h = 0,0,175,75

    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    number_of_whistles = setup()
    print(number_of_whistles)
    nowhistle_frame = 0
    whistle_frame = 0
    total_whistles = 0
    classes = {0: 'No Whistle',
            1: 'whistle'}
    np.set_printoptions(suppress=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    video_getter = VideoGet(source).start()
    ret,frame = video_getter.read()
    height, width, nchannels = frame.shape

    while ret:
        #if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            #video_getter.stop()
            
            #break


        size = (224, 224)
        image = cv2.resize(frame, size, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
#
    ##turn the image into a numpy array
        image_array = np.asarray(image)

#
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        #prediction = classes[google_model.predict_classes(data).item()]
        prediction = google_model.predict(data)
        probability = round(prediction[0][1]*100,2)

        if prediction[0][1] > 0.60:
            test = "I think this cooker is whistling with {}% Probability ".format(probability)
            whistle_frame = whistle_frame + 1

        else:
            nowhistleprobability = 100 - probability
            test = "I think this cooker is NOT whistling with {}% Probability ".format(nowhistleprobability)
            nowhistle_frame = nowhistle_frame + 1
            whistle_frame = 0
        if nowhistle_frame > 10:
            if whistle_frame > 15:
                total_whistles = total_whistles + 1
                if total_whistles < number_of_whistles :
                    say = str(total_whistles)+" whistles done"
                    print(say)
                #save_audio(say)
                    Thread(target=save_audio, args=(say,whistle_frame)).start()
                #Thread(target=play_audio, args=(local_ip,fname,cast)).start()
                #pool.apply_async(play_audio, [local_ip,fname,cast], callback)
                #loop.run_until_complete(play_audio(local_ip,fname,cast))

                nowhistle_frame = 0
                whistle_frame = 0
        text = " Whistle count: {}".format(total_whistles)
        if total_whistles >= number_of_whistles:
            say = str(total_whistles)+" whistles done.Turn off the gas"
            Thread(target=save_audio, args=(say,whistle_frame)).start()



            #Thread(target=play_audio, args=(local_ip,fname,cast)).start()
            #pool.apply_async(play_audio, [local_ip,fname,cast], callback)
            #loop.run_until_complete(play_audio(local_ip,fname,cast))

 


    # Capture frames in the video 
  
    # describe the type of font 
    # to be used. 
        font = cv2.FONT_HERSHEY_SIMPLEX 

     #Use putText() method for 
    #inserting text on video
        
        cv2.putText(frame,  
                    test,  
                    (50, 50),  
                    font, 1,  
                    (139,0,0),  
                    3,  
                    cv2.LINE_4) 
    
        
        cv2.putText(frame,  
                    text,  
                    (40, 100),  
                    font, 1.5,  
                    (139,0,0),  
                    3,  
                    cv2.LINE_4) 
    
     #Display the resulting frame 
        cv2.imshow('video', frame) 
    
        ret,frame = video_getter.read() 
    # creating 'q' as the quit  
    # button for the video 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
  
# release the cap object 
    
#out.release()
# close all windows 
    cv2.destroyAllWindows() #

        #cv2.imshow("Video", frame)
        
threadVideoGet(source="1.mp4")