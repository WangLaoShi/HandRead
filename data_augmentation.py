import random
import cv2
path = './' + 'test_image' + '/'

#旋转
def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90)#随机角度
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

if __name__ == "__main__":
    for i in range(5, 6):

        cnt = 21#计数
        for j in range(1, 21):
            roi = cv2.imread(path + str(i) + '_' + str(j)+'.png')
            for k in range(12):
                img_rotation = rotate(roi)#旋转
                cv2.imwrite(path + str(i) + '_' + str(cnt)+ '.png',img_rotation)
                cnt += 1
                img_flip = cv2.flip(img_rotation,1)#翻转
                cv2.imwrite(path + str(i) + '_' + str(cnt)+ '.png',img_flip)
                cnt += 1
            print(i,'_',j,'完成')

        '''
        roi = cv2.imread(path + str(i) + '_1.png')
        roi_flip = cv2.flip(roi,1)
        cv2.imwrite(path + str(i) + '_2.png',roi_flip)
        cnt = 3
        for k in range(9):
            img_rotation = rotate(roi)#rotation
            cv2.imwrite(path + str(i) + '_' + str(cnt)+ '.png',img_rotation)
            cnt += 1
            img_flip = cv2.flip(img_rotation,1)#flip
            cv2.imwrite(path + str(i) + '_' + str(cnt)+ '.png',img_flip)
            cnt += 1
        print(i,"完成")
        '''


