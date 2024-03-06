import cv2 as cv
import time
import os
import numpy as np
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import threading

model = r"models\haarcascade_frontalface_default.xml"
fr = cv.CascadeClassifier(model)
face_match = False

cap = cv.VideoCapture(0)
cap.set( cv.CAP_PROP_FRAME_WIDTH , 640 )
cap.set( cv.CAP_PROP_FRAME_HEIGHT, 480 )
cv.namedWindow('image')


if not cap.isOpened():
    print("Cannot open camera")
    cap.open()

# def mouse_event( event, x, y, flags, param):
#     pass
# def handle_change(x):
#     print(x)
# cv.setMouseCallback('image',mouse_event)
# cv.createTrackbar('R','image',0,255,handle_change)
# cv.createTrackbar('G','image',0,255,handle_change)
# cv.createTrackbar('B','image',0,255,handle_change)

counter = 0
while True:
    ret, frame = cap.read()

    if not ret : 
        print("Stream Error")
        time.sleep(1)
        break

    frame = cv.flip(frame, 1)

    # if counter %30 == 0:#limiting to once every 30 frame
    #     try :
    #         threading.Thread( target=check_face, args=(frame.copy(),) ).start()
    #     except ValueError:
    #         pass
    faces = None
    gray_img = cv.cvtColor( frame, cv.COLOR_RGB2BGR )
    cv.namedWindow("faces")
    for x, y, w, h in fr.detectMultiScale( gray_img, minNeighbors=10 ):
        cv.rectangle( frame,(x,y),(x+w,y+h),(0,0,255),1 )

        cropped_img = frame[ y:y+h, x:x+w :]
        cropped_img = cv.resize( cropped_img, (150,150))
        faces = np.vstack(( faces, cropped_img )) if not faces is None else cropped_img
        # break

    cv.imshow("image", frame )
    if not faces is None : cv.imshow('faces', faces )

    if cv.waitKey(1) == ord('q'):
        break
    counter += 1


cap.release()
cv.destroyAllWindows()
exit()