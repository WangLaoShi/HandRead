import cv2
import picture as pic
import classify as cf
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小

width, height = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
cnt = 1

cap = cv2.VideoCapture(0) #开摄像头

if __name__ == "__main__":
	while(1):
		flag, frame = cap.read() #读取摄像头的内容
		frame = cv2.flip(frame, 2)
		roi, res, ret, fourier_result, efd_result = pic.binaryMask(frame, x0, y0, width, height) #取手势所在框图并进行处理
		cv2.imshow("roi", roi)  # 显示手势框图
		cv2.imshow("res", res)
		cv2.imshow("ret", ret)
		key = cv2.waitKey(1) & 0xFF#按键判断并进行一定的调整
		#按'j''l''u''j'分别将选框左移，右移，上移，下移
		#按'q'键退出录像
		if key == ord('i'):
			y0 += 5
		elif key == ord('k'):
			y0 -= 5
		elif key == ord('l'):
			x0 += 5
		elif key == ord('j'):
			x0 -= 5
		elif key == ord('q'):
			break
		elif key == ord('s'):
			path ='./' + 'image_example' + '/'
			name = str(cnt)
			cv2.imwrite(path+'roi_sun.png',roi)
			cnt += 1
			#cv2.imwrite(path + name+'.png',ret)
		elif key == ord('p'):
			descirptor_in_use = abs(fourier_result)
			fd_test = np.zeros((1,31))
			temp = descirptor_in_use[1]
			for k in range(1,len(descirptor_in_use)):
				fd_test[0,k-1] = int(100 * descirptor_in_use[k] / temp)
			efd_test = np.zeros((1,15))
			for k in range(1,len(efd_result)):
				temp = np.sqrt(efd_result[k][0]**2 + efd_result[k][1]**2) + np.sqrt(efd_result[k][2]**2 + efd_result[k][3]**2)
				efd_test[0,k-1] = (int(1000 * temp))
			test_knn, test_svm = cf.test_fd(fd_test)
			print("test_knn =",test_knn)
			print("test_svm =",test_svm)
			test_knn_efd, test_svm_efd = cf.test_efd(efd_test)
			print("test_knn_efd =",test_knn_efd)
			print("test_svm_efd =",test_svm_efd)
		cv2.imshow('frame', frame) #播放摄像头的内容
	cap.release()
	cv2.destroyAllWindows() #关闭所有窗口


