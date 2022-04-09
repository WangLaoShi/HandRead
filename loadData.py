#!/usr/bin/env python
# -*-coding:utf-8 -*-
import efd
import fourierDescriptor as fd
import cv2
import numpy as np

path = './' + 'feature' + '/'
path_img = './' + 'image' + '/'

if __name__ == "__main__":
    for i in range(1, 11):
        for j in range(1, 201):
            roi = cv2.imread(path_img + str(i) + '_' + str(j) + '.png')

            descirptor_in_use = abs(fd.fourierDesciptor(roi))

            fd_name = path + str(i) + '_' + str(j) + '.txt'
            # fd_name = path + str(i) + '.txt'
            with open(fd_name, 'w', encoding='utf-8') as f:
                temp = descirptor_in_use[1]
                for k in range(1, len(descirptor_in_use)):
                    x_record = int(100 * descirptor_in_use[k] / temp)
                    f.write(str(x_record))
                    f.write(' ')
                f.write('\n')
            print(i, '_', j, '完成')
    '''
	for i in range(1, 11):
		for j in range(1, 21):
			roi = cv2.imread(path_img + str(i) + '_' + str(j) + '.png')
			efds, K, T = efd.elliptic_fourier_descriptors(roi, 16)
			efd_in_use = []
			for k in range(len(efds)):
				efd_in_use.append(np.sqrt(efds[k][0]**2 + efds[k][1]**2) + np.sqrt(efds[k][2]**2 + efds[k][3]**2))

			fd_name = path + str(i) + '_' + str(j) + '.txt'
			#fd_name = path + str(i) + '.txt'
			with open(fd_name, 'w', encoding = 'utf-8') as f:
				for k in range(1,len(efd_in_use)):
					x_record = int(1000*efd_in_use[k])
					f.write(str(x_record))
					f.write(' ')
				f.write('\n')
			print(i,'_',j,'完成')
	'''

