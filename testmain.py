
import cv2
from threading import Thread
from PIL import Image, ImageOps
import numpy as np  
from tensorflow import keras
from home import *
from configparser import ConfigParser
from VideoGet import VideoGet

net = cv2.dnn.readNet("yolov3_training_4000.weights", "yolov3_testing.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

classes = ["not_whisteling","whisteling"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

number_of_whistles = setup()

whistle_frame = 0
nowhistle_frame = 0
total_whistles = 0



video_getter = VideoGet().start()
img = video_getter.read()

height, width, channels = img.shape
font =  cv2.FONT_HERSHEY_DUPLEX 

while True:

    
    # Detecting objects

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)



    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.3:
                
                if int(class_id) == 1:
                    test = "I am {}% sure this cooker is Whisteling ".format(round(confidence*100,2))
                    whistle_frame = whistle_frame + 1
                else:
                    test = "I am {}% sure this cooker is NOT Whisteling ".format(round(confidence*100,2))
                    nowhistle_frame = nowhistle_frame + 1
                    whistle_frame = 0


                

                # Object detected
            
               
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                break
        else:
            continue
        break
    if nowhistle_frame > 5:
        if whistle_frame > 1:
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
            
            else:
                total_whistles >= number_of_whistles
                say = str(total_whistles)+" whistles done.Turn off the gas"
                Thread(target=save_audio, args=(say,whistle_frame)).start()
                nowhistle_frame = 0
                whistle_frame = 0
    text = "Whistle Count: {}".format(total_whistles)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            
            #frame = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[0], 2)
            #cv2.putText(img, label, (x, y + 30), font, 2, colors[0], 2)



    cv2.putText(img,  
                    test,  
                    (50, 50),  
                    font, 1,  
                    (255,255,255),  
                    2,  
                    cv2.LINE_4) 
    
        
    cv2.putText(img,  
                    text,  
                    (50, 100),  
                    font, 1.5,  
                    (255,255,255),  
                    2,  
                    cv2.LINE_4) 

    cv2.imshow("video",img)

    img = video_getter.read()

    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
        
cv2.destroyAllWindows() 
