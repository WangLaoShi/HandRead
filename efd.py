#!/usr/bin/env python
# -*-coding:utf-8 -*-
###计算椭圆傅里叶算子

import cv2
import numpy as np

def find_contours(Laplacian):
	#binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
	h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
	contour = h[1]
	contour = sorted(contour, key = cv2.contourArea, reverse=True)
	return contour


def deltaX_deltaY(res):

	gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
	Laplacian = cv2.convertScaleAbs(dst)
	contours_list = find_contours(Laplacian)
	deltaXY_list = []
	for contour in contours_list:
		contour = np.reshape(contour,[-1,2])
		dim0, dim1 = contour.shape
		difference = np.zeros((dim0, dim1))
		difference[0:dim0-1,:] = contour[1:dim0,:] - contour[0:dim0-1,:]
		difference[dim0-1,:] = contour[0,:] - contour[dim0-1,:]
		deltaX = - difference[:,0]
		deltaY = - difference[:,1]
		deltaXY_list.append((deltaX, deltaY))
	#print(deltaXY_list[0])
	return deltaXY_list


def elliptic_fourier_descriptors(bin_im,N = 16):

	deltaXY_list = deltaX_deltaY(bin_im)

	num_segments = len(deltaXY_list)

	efds= np.zeros((num_segments,N,4))

	K_array = np.zeros((num_segments))
	T_array = np.zeros((num_segments))

	counter = 0
	for deltaX, deltaY in deltaXY_list:
		efd, K, T = elliptic_fourier_descriptors_segment(deltaX,deltaY,N)
		efds[counter,:,:] = efd
		K_array[counter] = K
		T_array[counter] = T
		counter = counter + 1

	return efds[0], K_array, T_array


def elliptic_fourier_descriptors_segment(delta_X, delta_Y, N):

	efds = np.zeros((N,4))

	delta_t = np.sqrt(np.square(delta_X)+np.square(delta_Y))

	K = len(delta_X)

	t = np.cumsum(delta_t)
	T = np.sum(delta_t)

	if T == 0:      #just a single pixel segments... particular case
		return efds, K, T

	# there are N-1 a_n, b_n, c_n, d_n coefficients
	n_vector = np.arange(start=1,stop=N,step=1)
	n, p = np.meshgrid(n_vector, np.arange(K) )

	delta_xp = np.take(delta_X, p)
	delta_yp = np.take(delta_Y, p)
	delta_tp = np.take(delta_t, p)

	tp = np.take(t, p)

	tp_current = tp[0:K][:]
	tp_prev = np.vstack((tp[K-1][:],tp[0:K-1][:]))

	arg_grid1 = 2 * np.pi * n * tp_current / T
	arg_grid2 = 2 * np.pi * n * tp_prev / T

	cos_grid1 = np.cos(arg_grid1)
	cos_grid2 = np.cos(arg_grid2)

	sin_grid1 = np.sin(arg_grid1)
	sin_grid2 = np.sin(arg_grid2)

	factor = (T / (2 * np.pi**2 * np.square(n_vector)))

	Xfactor = (delta_xp/delta_tp)
	Yfactor = (delta_yp/delta_tp)

	prod = Xfactor * ( cos_grid1 - cos_grid2 )
	#a_n
	efds[1:,0] = factor * np.sum( prod , axis=0)

	prod = Xfactor * ( sin_grid1 - sin_grid2 )
	#b_n
	efds[1:,1] =  factor * np.sum( prod , axis=0)

	prod = Yfactor * ( cos_grid1 - cos_grid2 )
	#c_n
	efds[1:,2] = factor * np.sum( prod , axis=0)

	prod = Yfactor * ( sin_grid1 - sin_grid2 )
	#d_n
	efds[1:,3] = factor * np.sum( prod , axis=0)

	#A0, C0
	efds[0,0], efds[0,2] = continuous_components(delta_X = delta_X, delta_Y = delta_Y,
												 delta_t = delta_t, t = t, T = T, K = K )

	efds = rotation_and_scale_invariance(efds)

	return efds, K, T


def continuous_components(delta_X, delta_Y, delta_t, t, T, K):

	p = np.arange(K)

	delta_xp = np.take(delta_X, p)
	delta_yp = np.take(delta_Y, p)
	delta_tp = np.take(delta_t, p)
	tp = np.take(t, p)
	tp = np.hstack( ( np.array([0]) , tp ) )

	first_term_xi = np.cumsum(delta_X[0:K-1])
	second_term_xi = (delta_X[1:K]/delta_t[1:K]) * np.cumsum(delta_t[0:K-1])
	xi = np.hstack( ( np.array([0]), first_term_xi - second_term_xi ) )

	first_term_delta = np.cumsum(delta_Y[0:K-1])
	second_term_delta = (delta_Y[1:K]/delta_t[1:K]) * np.cumsum(delta_t[0:K-1])
	delta = np.hstack( ( np.array([0]), first_term_delta - second_term_delta ) )

	A0 = (1/T)*np.sum( (delta_xp/(2*delta_tp) * (np.square(tp[1:K+1]) - np.square(tp[0:K]))) + \
xi * (tp[1:K+1] - tp[0:K]))

	C0 = (1/T)*np.sum( (delta_yp/(2*delta_tp) * (np.square(tp[1:K+1]) - np.square(tp[0:K]))) + \
					   delta * (tp[1:K+1] - tp[0:K]))

	return A0, C0


def reconstruct(efds, T, K):

	T=np.ceil(T)

	N=len(efds)

	reconstructed = np.zeros((int(T),2))

	n = np.arange(start=1,stop=N,step=1)
	t = np.arange(T)

	n_grid, t_grid = np.meshgrid( n, t )

	a_n_grid = np.take(efds[:,0], n_grid)
	b_n_grid = np.take(efds[:,1], n_grid)
	c_n_grid = np.take(efds[:,2], n_grid)
	d_n_grid = np.take(efds[:,3], n_grid)

	arg_grid = n_grid * t_grid / T

	cos_term = np.cos( 2 * np.pi * arg_grid )
	sin_term = np.sin( 2 * np.pi * arg_grid )

	reconstructed[:,0] = efds[0,0] + np.sum(a_n_grid * cos_term + b_n_grid * sin_term, axis=1)
	reconstructed[:,1] = efds[0,0] + np.sum(c_n_grid * cos_term + d_n_grid * sin_term, axis=1)

	return reconstructed


def rotation_and_scale_invariance(efds):

	N = len(efds)

	a1 = efds[1,0]
	b1 = efds[1,1]
	c1 = efds[1,2]
	d1 = efds[1,3]

	theta1 = (1.0/2)*np.arctan( 2 * ( a1 * b1 + c1 * d1 ) / ( a1**2 + c1**2 - b1**2 - d1**2 ) )
	x11 = a1*np.cos(theta1) + b1*np.sin(theta1)
	y11 = c1*np.cos(theta1) + d1*np.sin(theta1)
	E0_1 = np.sqrt( np.square(x11) + np.square(y11) )

	theta2 = theta1 + np.pi / 2
	x22 = a1*np.cos(theta2) + b1*np.sin(theta2)
	y22 = c1*np.cos(theta2) + d1*np.sin(theta2)
	E0_2 = np.sqrt( np.square(x22) + np.square(y22) )

	if E0_1 >= E0_2:
		semimajor_axis = E0_1
		theta = theta1
	else:
		semimajor_axis = E0_2
		theta = theta2
	thetagrid = theta * np.arange(N)

	efds_star = efds.copy()

	efds_star[:,0] = np.cos(thetagrid) * efds[:,0] + np.sin(thetagrid) * efds[:,1]
	efds_star[:,1] = -np.sin(thetagrid) * efds[:,0] + np.cos(thetagrid) * efds[:,1]
	efds_star[:,2] = np.cos(thetagrid) * efds[:,2] + np.sin(thetagrid) * efds[:,3]
	efds_star[:,3] =  -np.sin(thetagrid) * efds[:,2] + np.cos(thetagrid) * efds[:,3]


	if (efds_star[1,2]!=0):
		if (efds_star[1,0]>0):
			phi1 = np.arctan(efds_star[1,2] / efds_star[1,0])
		else:
			phi1 = np.arctan(efds_star[1,2] / efds_star[1,0]) + np.pi
	else:
		if (efds_star[1,2]>0):
			phi1 = np.arctan(efds_star[1,2] / efds_star[1,0])
		else:
			phi1 = np.arctan(efds_star[1,2] / efds_star[1,0]) + np.pi

	alpha = np.cos(phi1)
	beta = np.sin(phi1)
	efds_2star = efds_star.copy()
	efds_2star[:,0] = (alpha * efds_star[:,0] + beta * efds_star[:,2]) / semimajor_axis
	efds_2star[:,1] = (alpha * efds_star[:,1] + beta * efds_star[:,3]) / semimajor_axis
	efds_2star[:,2] = (-beta * efds_star[:,0] + alpha * efds_star[:,2]) / semimajor_axis
	efds_2star[:,3] = (-beta * efds_star[:,1] + alpha * efds_star[:,3]) / semimajor_axis

	return efds_2star


###########测试#######
'''
if __name__ == "__main__":
	res = cv2.imread("test_image/1_1.png")
	test_efd, k, t = elliptic_fourier_descriptors(res, 4)
	print(test_efd)
	for i in range(4):
		efd_res = np.sqrt(test_efd[i][0]**2 + test_efd[i][1]**2) + np.sqrt(test_efd[i][2]**2 + test_efd[i][3]**2)
		print(efd_res)

'''
