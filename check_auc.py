
import numpy as np
import pickle
import pathlib as Path
import os.path
from timeit import default_timer as timer
import scipy.sparse
from sklearn.model_selection import GridSearchCV
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.svm import LinearSVC , NuSVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from os import listdir
from os.path import isfile,join

def plot_roc(roc_curve_values):
    np.save('./roc_curve_values.npy',roc_curve_values)
    plt.figure()
    for i in range(len(roc_curve_values)):
        plt.plot(roc_curve_values[i][0], roc_curve_values[i][1],
             lw=2, label='Round %d (auc = %0.4f)' % (i,roc_curve_values[i][2]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ')
    plt.legend(loc="lower right")
    # plt.savefig('/home/maryam/Code/python/adversarial_training/torch_impl/male_qabl/roc_all.eps'.format(round))

    plt.show()

    plt.close()

    plt.figure()
    plt.plot(roc_curve_values[0][0], roc_curve_values[0][1],
                 lw=2, label='Original (auc = %0.4f)' % roc_curve_values[0][2])
    plt.plot(roc_curve_values[-1][0], roc_curve_values[-1][1],
                 lw=2, label='Round 10 (auc = %0.4f)' % roc_curve_values[-1][2])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ')
    plt.legend(loc="lower right")
    plt.savefig('./roc_first_last.eps'.format(round))

    plt.show()

    plt.close()

x = scipy.sparse.load_npz('./MalwareDataset/x_test.npz').toarray()
y = np.load('./MalwareDataset/y_test.npy')
x_mal = x[np.where(y == 1)]
x_ben = x[np.where(y == 0)]
x_ben = x_ben[0:len(x_mal)]
y_mal = np.ones(x_mal.shape[0])
y_ben = np.zeros(x_ben.shape[0])
x = np.concatenate([x_mal, x_ben])
y = np.concatenate([y_mal, y_ben])
test_data = (x, y)

models_path = 'C:/Users/45028583/Desktop/Code/advGAN_pytorch/models/'
models = [f for f in listdir(models_path) if isfile(join(models_path,f))]
for file in models:
    model = pickle.load(open(models_path + file, 'rb'))
    predict = model.decision_function(x)
    fpr, tpr,_ = roc_curve(y, predict)
    roc_auc = auc(fpr, tpr)
    print(models_path + file)
    print('roc_auc: ', roc_auc)
    # print('accuracy_score: ', accuracy_score(y, predict))
    # print('precision_score: ', precision_score(y, predict))
    # print('recall_score: ', recall_score(y, predict))
    # print('f1_score: ', f1_score(y, predict))







# path=['./models/malJSAM-15/models/', './models/malJSAM-52/models/']
#
#
#
#
#
# for p in path:
#     for r in round:.
#
#
#
#         predict = model.decision_function(x)
#         fpr, tpr,_ = roc_curve(y, predict)
#         roc_auc = auc(fpr, tpr)
#         print(p+r)
#         print('roc_auc: ', roc_auc)
#         # print('accuracy_score: ', accuracy_score(y, predict))
#         # print('precision_score: ', precision_score(y, predict))
#         # print('recall_score: ', recall_score(y, predict))
#         # print('f1_score: ', f1_score(y, predict))