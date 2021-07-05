# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:40:07 2021

@author: sdaha
"""


############################################################
######    CHANGE the following things manually
############################################################

    ### Perspective transformation  ==> def pt()
    ### Laser locations  ==> def laser_loc(image)
    ### Lines to average for lane marking detection  ==> main loop

############################################################








import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
%matplotlib qt




### Convert image to Canny
def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur= cv2.GaussianBlur(gray,(5,5),0)
    blur= cv2.GaussianBlur(blur,(5,5),0)
    blur= cv2.GaussianBlur(blur,(9,9),0)
      
    low_threshold = 60 # original 40  # 10 works for persp ---- 20
    high_threshold =100 #original 150  # 60 works for persp -- 100
    canny = cv2.Canny(blur,low_threshold, high_threshold)
    return canny


'''
def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
    return line_image
'''


### Plot all the lines found from Hough lines
def plot_all_lines (image):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)
    return image


### Perspective transformation
def pt(image):
    width  = int(cap.get(3)) # float
    height = int(cap.get(4)) # float
    
    #Drawing Circles
    cv2.circle(frame,(650,550),5,(255,255,255), -1)
    cv2.circle(frame,(1450,550),5,(255,255,255), -1)
    cv2.circle(frame,(1700,800),5,(255,255,255), -1)
    cv2.circle(frame,(600,800),5,(255,255,255), -1)
    
    pts1 = np.float32([[650,550],[1450,550],[1700,800],[600,800]])
    pts2 = np.float32([[0,0],[1900,0],[1900,800],[0,800]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    result = cv2.warpPerspective(image, M, (1900,800))
    return result

'''
def big(image):
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(image,kernel,iterations = 1)
    opening = cv2.morphologyEx
    (dilation, cv2.MORPH_OPEN, kernel)
    return dilation, opening
'''

'''
def circ_PLOT(xx,yy, circles):
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
		
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle in the image
            # corresponding to the center of the circle
            cv2.circle(persp, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(persp, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            #print ("Column Number: ")
            #print (x)
            #print ("Row Number: ")
            #print (y)
            #print ("Radius is: ")
            print (r)
            xx.append(x)
            yy.append(y)
            xx_smooth= np.convolve(xx,np.ones(3),'valid')/3
            yy_smooth= np.convolve(yy,np.ones(3),'valid')/3
            plt.plot(xx,yy,':')
            plt.plot(xx_smooth,yy_smooth,linewidth = 5)
            plt.pause(0.001)
    return circles,counter
'''

### Draw location of laser and tentative tire
def laser_loc(image):
    ## This plots the location of fixed dot lasers projected on the ground from side mirror
    laser_gapX = 220        ### horizontal gap between two lasers  ## This equals 6 inch
    laser_gapy = -19        ### vertical gap between two lasers
    
    laser_init_x = 540      ### horizontal initial position of left laser
    laser_init_y = 525      ### vertical initial position of left laser
    
    ### Calculating the position of other three lasers
    laser1 = (laser_init_x + laser_gapX * 0, laser_init_y + laser_gapy * 0)
    laser2 = (laser_init_x + laser_gapX * 1, laser_init_y + laser_gapy * 1)
    laser3 = (laser_init_x + laser_gapX * 2, laser_init_y + laser_gapy * 2)
    laser4 = (laser_init_x + laser_gapX * 3, laser_init_y + laser_gapy * 3)
    
    ### approximate location of tire -- assuming 1 ft from left most laser. 
    tire = (laser_init_x - laser_gapX * 2, laser_init_y- laser_gapy * 2)
    ## Plot the locations on image
    cv2.circle(image,laser1,10,(0,0,0), -1)
    cv2.circle(image,laser2,10,(0,0,0), -1)
    cv2.circle(image,laser3,10,(0,0,0), -1)
    cv2.circle(image,laser4,10,(0,0,0), -1)
    cv2.circle(image,tire,20,(0,0,255), -1) ### APPROX Tire location
    return image
    
### Calculate moving average
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


### How many moving average
N=8


laser_gapX = 230



file = "01"
ext = ".MP4"

filename = file + ext
cap = cv2.VideoCapture(filename)

fps=cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(3))  # float `width`
height = int(cap.get(4))  # float `height`

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frame:",frame_total)
print("FPS:",fps)

#out_orig = cv2.VideoWriter(file+'_original.mp4', -1, fps, (width,height))
out_persp = cv2.VideoWriter(file+'_perspective.mp4', -1, fps, (1900,800))
out_canny = cv2.VideoWriter(file+'_canny.mp4', -1, fps, (width,height))


counter = 0
xx = []
yy = []
line_x1 = []
line_x2 = []
line_y1 = []
line_y2 = []

tire_x1 = []
tire_x2 = []
tire_y1 = []
tire_y2 = []
line_x1_all = []
line_x2_all = []
line_y1_all = []
line_y2_all = []

line_x1_avg = 0
line_x2_avg = 0
line_y1_avg = 0
line_y2_avg = 0


distance = []
time = []

line_image = np.zeros_like((800,1900,3),dtype=int)
all_line_image = np.zeros_like((800,1900,3),dtype=int)


plt.figure(1, figsize=(12, 24), dpi=80)
plt.legend()


while(cap.isOpened()):
    ret, frame = cap.read()
    cimg=frame
    
    
    
    if ret ==True:
               
########################################################################################################            

        ### Different images from the video
        
        canny_image = canny(frame)
        #bigger, bigger_open = big(canny_image)
        #persp_b = pt(bigger)
        persp_s = pt(canny_image)
        persp = pt(frame)
        
########################################################################################################        
        ### THIS is to find the right spot to find the lane markings
        ## This gives all the line images and circles to find which lines to average
        
        line_image = np.zeros_like(pt(frame))
        all_line_image = np.zeros_like(pt(frame))
        #line_image = np.zeros_like((800,1900,3),dtype=int)
        #all_line_image = np.zeros_like((800,1900,3),dtype=int)
        
        #To check the lines needed to be averaged
        cv2.circle(all_line_image,(1000,100),5,(255,255,255), -1)
########################################################################################################    
       
        ### Plot laser location (black) and tire location (red)
        laser_location = laser_loc(persp.copy())
########################################################################################################            
        
        ### HOUGH Circles to detect laser dot -- however, we don't need this as laser location is fixed.
        #circles =cv2.HoughCircles(persp_s, cv2.HOUGH_GRADIENT, 1, 100, param1=200, param2=5, minRadius=10, maxRadius=15)
########################################################################################################            

        ###  FIND hough lines        
        ## Used the perspective transform image so that it is smaller and only needed line is obtained
        lines = cv2.HoughLinesP(persp_s,3, np.pi/180, 800, np.array([]), minLineLength=200, maxLineGap=500)
        
########################################################################################################            
        ### Plot        
        persp_all_line = plot_all_lines(persp)        
########################################################################################################  

          
        ############ Detect lane markings and tire markings and make average
        if lines is None:
            print("Empty List")
            continue
        else:   
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                   
                
                ################################################################
                ################################################################
                #############  MANUALLY CHANGE THESE VALUES ####################
                ################################################################
                ################################################################
                
                if x1>700 and y1<100:      ############# Detecting lane markings 
                     line_x1.append(x1)
                     line_y1.append(y1)
                     line_x2.append(x2)
                     line_y2.append(y2)
                     
                if x1<300 and x1>100:      ############# Detecting tires
                     tire_x1.append(x1)
                     tire_y1.append(y1)
                     tire_x2.append(x2)
                     tire_y2.append(y2)
                     
                ################################################################
                ################################################################
                
                try:
                    
                    ## Average of two lane marking boundaries
                    #for some reason this didn't work --->   int(round(np.average(line_x1)))
                    line_x1_avg = int(round((min(line_x1)+max(line_x1))/2)) 
                    line_x2_avg = int(round((min(line_x2)+max(line_x2))/2))
                    line_y1_avg = int(round((min(line_y1)+max(line_y1))/2))
                    line_y2_avg = int(round((min(line_y2)+max(line_y2))/2))
                    
                    if line_x1_avg is not None: line_x1_all.append(line_x1_avg)
                    
                    ## Average of lines detected near tire (??)
                    tire_x1_avg = int(round(np.average(tire_x1)))
                    tire_x2_avg = int(round(np.average(tire_x2)))
                    tire_y1_avg = int(round(np.average(tire_y1)))
                    tire_y2_avg = int(round(np.average(tire_y2)))
                    
                except ValueError:
                    # if ValueError is raised, assign 0 to avgx1
                    if len(line_x1_all)>0: line_x1_avg = line_x1_all[-1]
                    if len(line_x2_all)>0: line_x2_avg = line_x2_all[-1]
                    if len(line_y1_all)>0: line_y1_avg = line_y1_all[-1]
                    if len(line_y2_all)>0: line_y2_avg = line_y2_all[-1]
                    
                    tire_x1_avg = 0
                    tire_x2_avg = 0
                    tire_y1_avg = 0
                    tire_y2_avg = 0
                
                
                line_image = np.zeros_like(persp)
                
                
                
                avg_TIRE = cv2.line(line_image,(tire_x1_avg,tire_y1_avg),(tire_x2_avg,tire_y2_avg),(200,155,100),10)
                avg_line_image = cv2.line(line_image,(line_x1_avg,line_y1_avg),(line_x2_avg,line_y2_avg),(255,155,155),10)
                
                midX= int(round((line_x1_avg+line_x2_avg)/2))
                midY= int(round((line_y1_avg+line_y2_avg)/2))
                cv2.circle(line_image,(midX,midY),20,(0,0,255), -1)
                
                midX_tire= int(round((tire_x1_avg+tire_x2_avg)/2))
                midY_tire= int(round((tire_y1_avg+tire_y2_avg)/2))
                cv2.circle(line_image,(midX_tire,midY_tire),20,(0,0,255), -1)
                
                
                
                
            distance.append((midX_tire-midX)/laser_gapX*6)
            time.append(counter/fps)
            AVG_dist= running_mean(distance,N)
            
            #plt.figure(1, figsize=(12, 6), dpi=80)
            plt.plot(distance,time,'k', label="Raw", linewidth= 1)
            if len(time[N-1:]) == len (AVG_dist):
                plt.plot(AVG_dist,time[N-1:],'b',label="Average of " + str(N), linewidth = 10, alpha = 0.4)
            #plt.legend()
            
            plt.ylabel("Time (sec)")
            plt.xlabel("Distance of tire from lane marking (inch)")
            plt.show()
            plt.pause(0.001)
            #plt.close(1)
              

            counter = counter + 1  ### To keep track of each frame to compute time

        
        #circ_PLOT(xx,yy, circles)
        
            
        #cv2.imshow('Perspective canny', persp_s)
        #cv2.imshow('Perspective', persp)
        cv2.imshow('Perspective + Laser position', laser_location)
        cv2.imshow('LINE',line_image)
        cv2.imshow('All lines- persp transform',persp_all_line)
        
        
        line_x1 = []
        line_x2 = []
        line_y1 = []
        line_y2 = []
        

             
        
        out_persp.write(persp_all_line)
        out_canny.write(canny_image)
        #out_orig.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        print("NO VID")
        break
plt.savefig(file + '_plot.png')
out_persp.release() 
#out_orig.release() 
out_canny.release() 

cap.release()
cv2.destroyAllWindows() 


'''
plt.figure(2)
plt.plot(distance)
plt.ylabel("Distance of tire from lane marking (inch)")
plt.xlabel("Time")
plt.show()
plt.pause(0.001)
'''
print("Total values plotted in the graph = ", len(distance))


################################################################
### Making CSV file
import csv
with open(file + '_distance.csv', 'w', newline = '') as csvfile:
    fieldnames = ['Time','Distance']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    thewriter.writeheader()
    for i in range(len(distance)):
        thewriter.writerow({'Time':time[i],'Distance':distance[i]})    
    
    