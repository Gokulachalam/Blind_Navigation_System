import streamlit as st
import cv2
import pyttsx3
import pytesseract
import keyboard
from collections import deque

thres = 0.45 # Threshold to detect object

engine = pyttsx3.init()

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

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

middle = deque(maxlen=10)

def object_in_middle(x, y, w, h):
    if (x + w/2) > 640/2 - 100 and (x + w/2) < 640/2 + 100:
        return True
    else:
        return False

def clear_middle():
    middle.clear()

def say_middle_names():
    if len(middle) > 0:
        str = ""
        for i in middle:
            str += i + "and"
        str = str[:-3]
        str += "at 12 o clock"
        engine.say(str)
        engine.runAndWait()

def read_text_box(x, y, w, h, img, name):
    img = img[y:y+h, x:x+w]
    text = pytesseract.image_to_string(img)
    if text == "":
        text = "no text"
    text = text+" written in "+name
    print(text)
    engine.say(text)
    engine.runAndWait()

def detect_objects(frame):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if object_in_middle(box[0], box[1], box[2], box[3]):
            cv2.rectangle(frame,box,color=(255,0,0),thickness=2)
            # cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            middle.append(classNames[classId-1].upper())
            # cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        else:
            cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            # cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

def read_text_in_objects(frame):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if object_in_middle(box[0], box[1], box[2], box[3]):
            read_text_box(box[0], box[1], box[2], box[3], frame, classNames[classId-1].upper())

def main():
    st.title("Object Detection")
    st.subheader("Press 'q' to quit")
    run = st.checkbox('Run')
    detect = st.checkbox('Detect')
    read = st.checkbox('Read_text written in objects')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        if detect:
            detect_objects(frame)
            say_middle_names()
        clear_middle()
        if read:
            read_text_in_objects(frame)
        FRAME_WINDOW.image(frame)
        st.write(middle)
        clear_middle()
        if keyboard.is_pressed('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

