import numpy as np 
import cv2
from filter import Kalman

def detect_object(img):
    # convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # detecting ball using canny edge detector
    edges = cv2.Canny(img,30,200)
    
    # thresolding image to two values. Those values above 254 is given as 255 and rest 0
    ret, edges = cv2.threshold(edges,254,255,cv2.THRESH_BINARY)
    
    # detect countors from this image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 3
    max_radius_thresh= 100   
    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        #Take only the valid circles
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
            
    return centers




KF = Kalman(dt=0.1, U=[1, 1], std=[0.1,0.1,1])
VideoCap = cv2.VideoCapture('ball.mp4')

def tracker():
    while(True):
        # Read frame
        ret, frame = VideoCap.read()
        centers = detect_object(frame)
        # If centroids are detected then track them
        if (len(centers) > 0):
            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 27, (0, 191, 255), 2)
        
            #predict the object
            (x,y) = KF.predict()
        
            cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), (255, 0, 0), 2)
            # Update
            (x1, y1) = KF.update(centers[0])
            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (x1 - 30, y1 - 30), (x1 + 30, y1 + 30), (0, 0, 255), 2)
        
            cv2.putText(frame, "Final Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (centers[0][0] + 15, centers[0][1] - 15), 0, 0.5, (0,191,255), 2)
        
            cv2.imshow('image', frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                VideoCap.release()
                cv2.destroyAllWindows()
                break    
            
            cv2.waitKey(50)



if __name__ == "__main__":
    tracker()