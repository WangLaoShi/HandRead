#!/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

path = './' + 'feature' + '/'
model_path = "./model/"
test_path = "./test_feature/"

k_range = range(1, 21)
test_accuracy = []


def txtToVector(filename, N):
	returnVec = np.zeros((1,N))
	fr = open(filename)
	lineStr = fr.readline()
	lineStr = lineStr.split(' ')
	for i in range(N):
		returnVec[0, i] = int(lineStr[i])
	return returnVec

def tran_KNN(N, k):
	hwLabels = []
	trainingFileList = listdir(path)
	m = len(trainingFileList)
	trainingMat = np.zeros((m,N))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = txtToVector(path+fileNameStr, N)
	print("k=%d训练结束"%k)
	neigh = KNN(n_neighbors=k, algorithm= 'auto')
	neigh.fit(trainingMat,hwLabels)
	print("KNN Model save...")
	save_path = model_path + "knn_efd_" +str(k)+ "_train_model.m"
	joblib.dump(neigh,save_path)

def test_KNN(neigh, N):
	testFileList = listdir(test_path)
	errorCount = 0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNum = int(fileNameStr.split('_')[0])
		vectorTest = txtToVector(test_path+fileNameStr,N)
		valTest = neigh.predict(vectorTest)
		#print("分类返回结果为%d\t真实结果为%d" % (valTest, classNum))
		if valTest != classNum:
			errorCount += 1
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
	test_accuracy.append((200 - errorCount)/ 200.0)

def tran_SVM(N):
	svc = SVC()
	parameters = {'kernel':('linear', 'rbf'),
	              'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
	              'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}#预设置一些参数值
	hwLabels = []#存放类别标签
	trainingFileList = listdir(path)
	m = len(trainingFileList)
	trainingMat = np.zeros((m,N))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = txtToVector(path+fileNameStr,N)#将训练集改为矩阵格式
	print("数据加载完成")
	clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)#网格搜索法，设置5-折交叉验证
	clf.fit(trainingMat,hwLabels)
	print(clf.return_train_score)
	print(clf.best_params_)#打印出最好的结果
	best_model = clf.best_estimator_
	print("SVM Model save...")
	save_path = model_path + "svm_efd_" + "train_model.m"
	joblib.dump(best_model,save_path)#保存最好的模型

def test_SVM(clf,N):
	testFileList = listdir(test_path)
	errorCount = 0#记录错误个数
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNum = int(fileNameStr.split('_')[0])
		vectorTest = txtToVector(test_path+fileNameStr,N)
		valTest = clf.predict(vectorTest)
		#print("分类返回结果为%d\t真实结果为%d" % (valTest, classNum))
		if valTest != classNum:
			errorCount += 1
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))


def test_fd(fd_test):
	neigh = joblib.load(model_path + "knn_7_train_model.m")
	test_knn = neigh.predict(fd_test)
	clf = joblib.load(model_path + "svm_train_model.m")
	test_svm = clf.predict(fd_test)
	return test_knn, test_svm

def test_efd(efd_test):
	neigh = joblib.load(model_path + "knn_efd_7_train_model.m")
	test_knn_efd = neigh.predict(efd_test)
	clf = joblib.load(model_path + "svm_efd_train_model.m")
	test_svm_efd = clf.predict(efd_test)
	return test_knn_efd, test_svm_efd

####训练 + 验证#####
'''
if __name__ == "__main__":
	for i in range (1, 21):
		tran_KNN(15, i)
		neigh_efd = joblib.load(model_path + "knn_efd_"+str(i) + "_train_model.m")
		test_KNN(neigh_efd, 15)
	plt.plot(test_accuracy,'r-o')
	plt.title("Testing Accuracy of KNN")
	plt.xlabel("Value of K for KNN")
	plt.ylabel("Testing Accuracy")
	plt.savefig("knn_efd")
	tran_SVM(15)
	clf = joblib.load(model_path + "svm_efd_" + "train_model.m")
	test_SVM(clf,15)
'''



