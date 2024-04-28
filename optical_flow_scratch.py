from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import time
import math
import cv2

def grayimage(input_img1):
	img1G = cv2.cvtColor(input_img1,cv2.COLOR_BGR2GRAY)
	return img1G
def smoothimage(input_img1):
	img1G = cv2.GaussianBlur(input_img1,(3,3),0)
	return img1G
def calculatevelocities(feature,h,w,Ix,Iy,It):
	u = np.nan*np.ones((h,w))
	v = np.nan*np.ones((h,w))
	for l in feature:
		j,i = l.ravel()
		IX,IY,IT = [],[],[]
		if(i+2 < h and i-2 > 0 and j+2 < w and j-2 > 0):
			for b1 in range(-2,3):
				for b2 in range(-2,3):
					IX.append(Ix[i+b1,j+b2])
					IY.append(Iy[i+b1,j+b2])
					IT.append(It[i+b1,j+b2])
			LucasK = (IX,IY)
			LucasK = np.matrix(LucasK)
			LucasK_T = np.array(np.matrix(LucasK))
			LucasK = np.array(np.matrix.transpose(LucasK)) 
			
			A1 = np.dot(LucasK_T,LucasK)
			A2 = np.linalg.pinv(A1)
			A3 = np.dot(A2,LucasK_T)
			(u[i,j],v[i,j]) = np.dot(A3,IT)
	return u,v	
def plotarrow(h,w,u,v,colorImage1):
	fig = plt.figure('')
	plt.subplot(1,1,1)
	plt.axis('off')
	plt.imshow(colorImage1, cmap = 'gray')
	for i in range(h):
		for j in range(w):
			if abs(u[i,j]) > t or abs(v[i,j]) > t:
				plt.arrow(j,i,1.5*(-1*u[i,j]),1.5*(-1*v[i,j]), head_width = 3, head_length = 3, color = 'red')
	return fig
def opticalFlow(input_img1,input_img2,frameIndex,totalFrames):
	h,w = input_img1.shape[:2]
	colorImage1 = cv2.cvtColor(input_img1,cv2.COLOR_BGR2RGB)
	img1G = grayimage(input_img1)
	img2G = grayimage(input_img2)
	img1 = np.array(img1G)
	img2 = np.array(img2G)
	smooth_img1 = smoothimage(img1)
	smooth_img2 = smoothimage(img2)
	# Gradient calculation, spationl and temporal
	Ix = signal.convolve2d(smooth_img1,[[-0.25, 0.25],[-0.25, 0.25]],'same') + signal.convolve2d(smooth_img2,[[-0.25, 0.25],[-0.25, 0.25]],'same')
	Iy = signal.convolve2d(smooth_img1,[[-0.25,-0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(smooth_img2,[[-0.25,-0.25],[ 0.25, 0.25]],'same')
	It = signal.convolve2d(smooth_img1,[[ 0.25, 0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(smooth_img2,[[-0.25,-0.25],[-0.25,-0.25]],'same')
	# feature calculation
	features = cv2.goodFeaturesToTrack(smooth_img1,10000,0.01,10)	
	feature = np.int0(features)
	u,v=calculatevelocities(feature,h,w,Ix,Iy,It)
	
	# show the gram image
	fig=plotarrow(h,w,u,v,colorImage1)
	print('\r({:4}/{:4}) - Time Elapsed: {:10.10} seconds'.format(frameIndex+1,totalFrames,time.time()-start), end='')
	# showing and converting to original format
	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	plt.close()

	return img


start = time.time()
t = 0.7
video = cv2.VideoCapture(0)
totalnumberofFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

_,image1 = video.read()
ret,image2 = video.read()
imageCounter = 0

videoName = 'optical_flow.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
first = True

while(ret):
	image = opticalFlow(image1,image2,imageCounter,totalnumberofFrames)
	
	if(not imageCounter):
		height,width = image.shape[:2]
		videoOut = cv2.VideoWriter(videoName, fourcc, fps, (width,height))

	videoOut.write(image)
	cv2.imshow("Optical Flow",image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	
	image1 = image2.copy()
	ret,image2 = video.read()
	imageCounter += 1

print("")
video.release()
videoOut.release()