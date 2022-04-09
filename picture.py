import cv2
import numpy as np
import math
import fourierDescriptor as fd
import efd as efd
from scipy.ndimage.interpolation import geometric_transform

MIN_DESCRIPTOR = 500
NUMBER_OF_HARMONICS = 20

#二值化处理
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0)) #画出截取的手势框图
	roi = frame[y0:y0+height, x0:x0+width] #获取手势框图
	#roi = topolar(roi)
	#cv2.imshow("roi", roi) #显示手势框图
	res = skinMask(roi) #进行肤色检测
	#cv2.imshow("res", res) #显示肤色检测后的图像
	'''
	kernel = np.ones((3,3), np.uint8) #设置卷积核
	res = cv2.erode(res, kernel) #腐蚀操作
	res = cv2.dilate(res, kernel)#膨胀操作
	cv2.imshow("res", res) #显示肤色检测后的图像
	'''

	ret, fourier_result = fd.fourierDesciptor(res)
	efd_result, K, T = efd.elliptic_fourier_descriptors(res)
	#ret = fd.reconstruct(res, fourier_result, MIN_DESCRIPTOR)
	return roi, res, ret, fourier_result, efd_result



#肤色检测
'''
##########方法一###################
##########BGR空间的手势识别#########
def skinMask(roi):
	rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) #转换到RGB空间
	(R,G,B) = cv2.split(rgb) #获取图像每个像素点的RGB的值，即将一个二维矩阵拆成三个二维矩阵
	skin = np.zeros(R.shape, dtype = np.uint8) #掩膜
	(x,y) = R.shape #获取图像的像素点的坐标范围
	for i in range(0, x):
		for j in range(0, y):
			#判断条件，不在肤色范围内则将掩膜设为黑色，即255
			if (abs(R[i][j] - G[i][j]) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j]):
				if (R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) \
						and (max(R[i][j],G[i][j],B[i][j]) - min(R[i][j],G[i][j],B[i][j]) > 15):
					skin[i][j] = 255
				elif (R[i][j] > 220) and (G[i][j] > 210) and (B[i][j] > 170):
					skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin) #图像与运算
	return res
'''

'''
##########方法二###################
########HSV颜色空间H范围筛选法######
def skinMask(roi):
	low = np.array([0, 48, 50]) #最低阈值
	high = np.array([20, 255, 255]) #最高阈值
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #转换到HSV空间
	mask = cv2.inRange(hsv,low,high) #掩膜，不在范围内的设为255
	res = cv2.bitwise_and(roi,roi, mask = mask) #图像与运算
	return res
'''

'''
##########方法三###################
#########椭圆肤色检测模型##########
def skinMask(roi):
	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,Cr,Cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(Cr.shape, dtype = np.uint8) #掩膜
	(x,y) = Cr.shape
	for i in range(0, x):
		for j in range(0, y):
			if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #若不在椭圆区间中
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res
'''


################方法四####################
####YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res


'''
##########方法五###################
########Cr，Cb范围筛选法###########
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(cr.shape, dtype = np.uint8)
	(x,y) = cr.shape
	for i in range(0, x):
		for j in range(0, y):
			#每个像素点进行判断
			if(cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res
'''
