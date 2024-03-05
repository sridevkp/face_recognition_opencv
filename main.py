import cv2 as cv
import time
from deepface import DeepFace
import threading

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

def check_face(frame):
    print("thread")

counter = 0
while True:
    ret, frame = cap.read()

    if not ret : 
        print("Stream Error")
        time.sleep(1)
        break

    frame = cv.flip(frame, 1)

    if counter %30 == 0:#limiting to once every 30 frame
        try :
            threading.Thread( target=check_face, args=(frame.copy(),) ).start()
        except ValueError:
            pass

    if face_match : 
        # display result
        pass

    cv.imshow('image',frame)
    
    if cv.waitKey(1) == ord('q'):
        break
    counter += 1


cap.release()
cv.destroyAllWindows()
exit()