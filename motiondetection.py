# Importing Libraries
import cv2
import numpy as np
from flask import Flask, render_template, Response

drawing = False
point1 = ()
point2 = ()

###### Function for drawing rectangle using laptop #####

def mouse_drawing(event,x,y,flags,params):
    global drawing,point1,point2
    
    # when left button of mouse is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            drawing = True
            point1 = (x,y)
        else:
            drawing = False
    # moving mouse 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            point2 = (x,y)
            
def point_is_within_limits(pt, minX, minY, maxX, maxY):
    """Check single given point if it is within the given limits"""
    if len(pt) == 2:
        if pt[0] >= minX and pt[0] <= maxX and pt[1] >= minY and pt[1] <= maxY:
            return True
    return False
 
def count_points_within_limits(points, minX, minY, maxX, maxY):
    """Goes through all points in list and checks how many of them within limits"""
    within_limits = 0
    for pt in points:
        if point_is_within_limits(pt, minX, minY, maxX, maxY):
            within_limits += 1
    return within_limits
 
def get_corner_points(x1, y1, w, h):
    """Returns 4 corner points, top-left, top-right, bottom-right, bottom-left"""
    return (x1, y1), (x1+w, y1), (x1+w, y1+h), (x1, y1+h)
 
def get_points_limits(point1, point2):
    """Get min and max coordinates of points"""
    minX = maxX = minY = maxY = 0
    if len(point1) == 2 and len(point2) == 2: #if exactly two coordinates present
        minX = min(point1[0], point2[0])
        minY = min(point1[1], point2[1])
        maxX = max(point1[0], point2[0])
        maxY = max(point1[1], point2[1])
    return minX, maxX, minY, maxY
 
def set_rectangle_limits(x1, y1, x2, y2, minX, minY, maxX, maxY):
    """Crop rectangle to fit given limits"""
    if len(point1) == 2 and len(point2) == 2: #if exactly two coordinates present
        x1 = min(max(x1, minX), maxX)
        y1 = min(max(y1, minY), maxY)
        x2 = max(min(x2, maxX), minX)
        y2 = max(min(y2, maxY), minY)
    return x1, y1, x2, y2

app = Flask(__name__)



@app.route('/') # tells app that index() should reun when app receives the data from client
def index():
    # we will stream our video on our index.html file
	return render_template('index.html')

def gen():
    
    """Capturing the Video/Frames"""
    cap = cv2.VideoCapture(0)
    """Creating the foreground background mask"""
    video = cv2.createBackgroundSubtractorMOG2(history = 300, varThreshold = 30, detectShadows = False)
    cv2.namedWindow("Output")
    cv2.setMouseCallback("Output",mouse_drawing)
    """Codec for saving video"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    """Writer for saving video"""
    out = cv2.VideoWriter("Thesis_0.mp4",fourcc,10,(640,480))
    i = 0
    
    while True:
        i += 1
        # reading frame
        ret,frame = cap.read()
        # resizing the frame
        frame_r = cv2.resize(frame, (640,480))
        # applying fgbg mask on the frames
        mask = video.apply(frame_r, learningRate = 0.003)
        
        
        no_pixels = mask.shape[0]*mask.shape[1]
        no_black_pixels = np.abs(cv2.countNonZero(mask) - no_pixels)
        
        # we will try to save video/frames according to our thresh1
        # basically used for saving only frames with motion
        thresh1 = 0.93
        x = int(thresh1*no_pixels)
        
        # for drawing rectangle using mouse
        if point1 and point2:
            cv2.rectangle(frame_r,point1,point2,(0,0,255),0)
          
        # applying binary thresholding and dilation 
        # dilate to fill the tiny holes in the frame.
        # Dilation grow,expand the effect on a binary
        # will make it easy to find contours
        thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
        dilate = cv2.dilate(thresh.copy(),None,iterations = 3)
        
        # finding contours
        contours,hier = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # if area of contour is less than 5000
            if cv2.contourArea(contour) < 5000:
                continue
            (x1,y1,w,h) = cv2.boundingRect(contour)
            #get corners to check if at least one point is within given limits
            p1,p2,p3,p4 = get_corner_points(x1,y1,w,h)
            minX, maxX, minY, maxY = get_points_limits(point1, point2)
            if count_points_within_limits([p1, p2, p3, p4], minX, minY, maxX, maxY) > 0:
                #crop rectangle to fit into given rectangle
                x1, y1, x2, y2 = (x1, y1, x1+w, y1+h)
                x1, y1, x2, y2 = set_rectangle_limits(x1, y1, x2, y2, minX, minY, maxX, maxY)
                cv2.rectangle(frame_r,(x1,y1),(x2,y2),(0,255,0),2)
                
                # save frames with foreground 
                # according to thresh1 value
                # In simple words, save frames/video with motion only
                if no_black_pixels <= x:
                    print('Save')
                    cv2.imwrite('frame%d.jpg' %i,frame_r)
                    out.write(frame_r)
                
                #The function imencode compresses the image 
                #and stores it in the memory buffer 
                #that is resized to fit the result
                frame = cv2.imencode('.jpg',frame_r)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
          
        # show stream(Python)
        cv2.imshow("Output",frame_r)
        
        # press escape to stop recodring
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    


@app.route('/video_feed')
def video_feed():
    # or the purpose of having a stream 
    # each part should replace the previous part
    # multipart/x-mixed-replace is used for that.
    return Response(gen(),mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
