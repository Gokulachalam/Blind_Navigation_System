import cv2
import pyttsx3
import pytesseract
thres = 0.45 # Threshold to detect object

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
cap = cv2.VideoCapture(0)
middle = []

def object_in_middle(x, y, w, h):
    if (x + w/2) > 640/2 - 100 and (x + w/2) < 640/2 + 100:
        return True
    else:
        return False

def clear_middle():
    middle.clear()

def say_middle_names():
    str = ""
    for i in middle:
        str += i + "and"
    str = str[:-3]
    str += "at 12 o clock"
    engine.say(str)
    engine.runAndWait()

def read_text_box(x, y, w, h, img, name):
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    img = img[y:y+h, x:x+w]
    text = pytesseract.image_to_string(img)
    if text == "":
        text = "no text"
    text = text+" written in "+name
    print(text)
    engine.say(text)
    engine.runAndWait()


while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres) 

    if len(classIds) != 0:
        k = cv2.waitKey(33)
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if object_in_middle(box[0], box[1], box[2], box[3]):
                cv2.rectangle(img,box,color=(255,0,0),thickness=2)
                # cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                if classNames[classId-1] not in middle:
                    middle.append(classNames[classId-1])
                print(middle)
                if k == ord('r'):
                    cr = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    cv2.imshow("cropped", cr)
                    read_text_box(box[0], box[1], box[2], box[3], img, classNames[classId-1])
                
                if k == ord('d'):
                    focal_length = (box[2] * 3.04) / 4
                    distance = (3.04 * 720) / (box[2] * 2.54)
                    print("Distance: ", distance)
                    engine.say(classNames[classId-1]+" at distance: "+str(round(distance,2))+ " centimeters")
                    engine.runAndWait()

            else:
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                # cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        if k==ord('h'): 
            say_middle_names()
            clear_middle()
        clear_middle()
        if k==ord('q'):
            break

    cv2.imshow("Output",img)
    cv2.waitKey(1)

