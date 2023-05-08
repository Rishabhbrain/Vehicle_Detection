import cv2
import numpy as np

#strating web camera
capture = cv2.VideoCapture(r"C:\Users\risha\PycharmProjects\pythonProject\video.mp4")
min_width_react = 80
min_height_react = 80
#capturing frame through camera and displaying

#Initialize substrator
algo = cv2.createBackgroundSubtractorMOG2()


def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
# the above algorithm is used to substract the vehicle from the video using backgrounds.

count_line_position = 550;
while True:
    ret, frame1 = capture.read()


    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernal)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernal)
    countershape,h = cv2.findContours(dilatada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200, count_line_position),(255,127,0),3)
#    cv2.imshow('Detector', dilatada)



    for (i,c)in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter =(w>= min_width_react) and (h>= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0,0,255),-1)



    cv2.imshow('Video OG', frame1)

    if cv2.waitKey(1) == 13: #click enter to exit
        break

cv2.destroyAllWindows()
capture.release()

