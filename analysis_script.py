#use python 3.6 - run from fast.ai environment
#!home/llu-2/anaconda3/envs/fastai/bin/python

import os
from fastai.vision import *
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
from statsmodels.tsa.filters import hp_filter
from natsort import natsorted, ns
import time

#path to all
path = Path('analysis')

#path for non live testing
#picture_path = Path('clotting_test_2')

#path for live test
picture_path= Path('cropped_pics')

#path_to_IMG = Path(/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/cropped)


classes = ['clot', 'non_clot']

full_clot_probability_array = []
frame_number = 1

full_path = Path("analysis/cropped_pics")

endpoint = []
endpoint_final_frame_number = []
				
#for filename in sorted(os.listdir(path/picture_path), key=lamba)
for filename in natsorted(os.listdir(path/picture_path)):
	
	
    img = open_image(path/picture_path/filename)

    learn = load_learner(path, file='resnet34_02_error_Sept_19_with_transforms_rectangles.pkl')
  
    pred_class,pred_idx,outputs = learn.predict(img)
    print(filename)

	#pred_class2,pred_idx2,outputs2 = learn.predict(imgM2)	
	
	#this_image = os.path.join('/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics', filename)
    this_image = os.path.join('/home/chris/Desktop/blood_data/analysis/cropped_pics', filename)	
    image_read = cv2.imread(this_image)
	#plt.imshow(image_read, interpolation = 'bicubic')
	#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	#plt.show()
	#plt.close()
	
	#resized_image = cv2.resize(image_read, (1800, 400))	
	#cv2.imshow('tube', resized_image)	
	#cv2.waitKey(1)

	#store probabilities, when probability of 4/5 frames is over 80% a clot, call it at that 		time; first number is prob clot, second is prob non_clot		
	#take last five (output values), add all up, / by 5, get at least 80%, then output that 	time	
	
	#convert torch.Tensor to nparray
    print(type(outputs))
    tensor_to_np = outputs.numpy()
    print(tensor_to_np)

	#grab clot probability from each output
    grab_clot_probability = tensor_to_np[0]
    print(grab_clot_probability)
	
	#add each new clot probability to end of array
	
    full_clot_probability_array.append(grab_clot_probability)
    print("probability array is")
    print(full_clot_probability_array)	    
    
    plt.plot(full_clot_probability_array)	
    plt.draw()
    plt.pause(.001)	

	#take last 5 clot probabilities in this array, add them up, divide by 5, if equal over .8, call a clot
	#last_five_array = full_clot_probability_array[-5:]
	#sum_last_five = sum(last_five_array)
	#last_five_avg = sum_last_five/5 
	#last_five_avg = float(last_five_avg)
	#print("avg is" + str(last_five_avg))

	

	#if last_five_avg > .8:
	#    print("the blood has clotted")		
	#else:
	#    print("the blood has not clotted")	

	#smooth graph/remove noise with hp filter; store the smoothed values in "trend", and plot on same graph
    if frame_number > 1:
		#cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=50)
		#cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=5800)
		#cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=50000)
        cycle, trend = hp_filter.hpfilter(full_clot_probability_array, lamb=150000)
        print("trend is")		
        print(trend)		
        plt.plot(trend)		
        plt.draw()
        plt.pause(.001)
        plt.clf()
		
	
	
	#to pick endpoint, find where the first gradient trend is positive (>.25) for certain number of frames (50 for now)
	#if frame_number > 482: 
	#	#i_max = np.argmax(np.abs(np.gradient(trend, 50)))
	#	#i_max = np.argmax(np.gradient(trend, 50))
	#	endpoint = [None]
	#	print("argmax gradient is")
	#	print(np.argmax(np.gradient(trend, 50)))		
	#	if not endpoint:
	#		if (np.argmax((np.gradient(trend, 50)) > 0.25)):
	#			endpoint = (np.argmax((np.gradient(trend, 50)) > 0.25))
	#	#print(i_max)
	#	print(endpoint)		
	#	#print(frame_number[i_max]) 
		
	#write a function that takes numbers in trend array(iterates through them), takes first number and a number 50 after, and checks if it is .2 higher
    print("frame number is")		
    print(frame_number)		
    if frame_number > 50:
        print("in loop")
        print("trend[frame_number] is")
        print(trend[frame_number-1])

        
        #check if endpoint array is empty and if slope is greater than ___ from this frame to 50 frames back	
        if not endpoint and ((trend[frame_number-1]-trend[frame_number-50]) > .3): 							
            print("in endpoint loop")
            print(trend[frame_number-1])
            #append slope to endpoint list  				
            endpoint.append(trend[frame_number-1])
            endpoint_final_frame_number = frame_number - 50            

    print("endpoint is")	
    print(endpoint)
	
    #plot endpoint line on graph
    if endpoint:
        print("endpoint frame is")  
        print(endpoint_final_frame_number)       
        plt.axvline(x=endpoint_final_frame_number, color='r', linestyle='--')



    #polyfit algorithm	
	#y = full_clot_probability_array
	#x = []
	
	#for i in range(0, len(full_clot_probability_array)):
	#	x.append(i)
  
	#do polyfit, print out polyfit values and corresponding frame for that degree 
	#if count>10:
		
	#	p1 = np.polyfit(x,y,3)
	#	print(p1)
	#	#plt.plot(x,y,'o')
	#	plt.plot(x, np.polyval(p1,x),'r-')
	

	#return pred_class1, pred_class2
	
	#print(pred_class)
	#print(pred_idx)
    print(outputs)
	#print(pred_class2)
	#print(clot_probability)

    frame_number = frame_number + 1	
