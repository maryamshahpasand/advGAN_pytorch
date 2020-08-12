import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.utils import deprecated
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm, tree
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import pathlib as Path
import os.path
from timeit import default_timer as timer
import scipy.sparse
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.svm import LinearSVC , NuSVC
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x

class surrogate_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(surrogate_model, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, hidden_size)


        self.map5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        x = torch.sigmoid(self.map3(x))
        x = torch.sigmoid(self.map4(x))

        # x= F.elu(self.map3(x))
        # x= torch.sigmoid(self.map3(x))
        return torch.softmax(self.map5(x), dim=1)
        # return torch.sigmoid(self.map3(x))

class malware_classifier_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(malware_classifier_net, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        # x= F.elu(self.map3(x))
        # x= torch.sigmoid(self.map3(x))
        return torch.softmax(self.map3(x), dim=1)
        # return torch.sigmoid(self.map3(x))

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

class Mal_Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mal_Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.map1(x))
        return F.relu(self.map2(x))


class Mal_Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mal_Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = torch.cat((x_and_example[0], x_and_example[1]), 1)
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        x = F.relu(self.map3(x))

        # return torch.max(x_and_example[0], x)
        return x
# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class malware_classifeir():
    def __init__(self , type = 'RF' , input_size = 0, hidden_size= 0, output_size  = 0):
        self.type = type
        if 'RF' in self.type:
            Parameters={'n_estimators':[100], 'max_depth':[50]}
            self.model = GridSearchCV(RandomForestClassifier(), Parameters, cv=5, scoring='f1', n_jobs=-1)
        elif 'SVM' in self.type:
            # model = svm.SVC()
            # model = svm.SVC(kernel='linear', class_weight={1: 9})
            # C_range = np.logspace(-2, 10, 13)
            # gamma_range = np.logspace(-9, 3, 13)
            # param_grid = dict(gamma=gamma_range, C=C_range)
            # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            # self.model = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

            Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            self.model = GridSearchCV(LinearSVC(), Parameters, cv=5, scoring='f1', n_jobs=-1)
        elif 'RBF_SVM'  in self.type:
            # Parameters={'kernel':['rbf'],'gamma': [1e-3, 1e-4],'C': [0.001, 0.01, 0.1]}
            # model = GridSearchCV(svm.SVC(), Parameters, cv=5, scoring='f1', n_jobs=-1)
            self.model = svm.SVC(kernel='rbf')
        elif 'LR' in self.type:
            self.model = linear_model.LogisticRegression()
        elif 'DT' in self.type:
            self.model = tree.DecisionTreeRegressor()
        elif 'KNN' in self.type:
            self.model = KNeighborsClassifier()
        elif 'MLP' in self.type:
            self.model = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=200, alpha=1e-4,
                                  solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                  learning_rate_init=.1)
        elif 'DNN' in self.type:
            self.model = malware_classifier_net(input_size, hidden_size, output_size)
    def train(self ,data,test_data ):
        if 'NN' in self.type:
            print('yes')
        #     train_malware_classifier_net(malware_classifier_net, dataloader, epochs,device)
        else:
            (xmal, ymal), (xben, yben) = data[0], data[1]
            xtrain = np.concatenate([xben, xmal])
            ytrain = np.concatenate([yben, ymal])
            # print('training set shape:', xtrain.shape[0])
            # print('\n---------------------')

            start_train = timer()

            # self.model = pickle.load(open('./models/generalizability/MALGAN_all_SVM_R10.sav', 'rb'))
            if os.path.isfile('./models/' + self.type + '.sav'):
                # print('LOADING the ' + self.type + ' CLASSIFIER...\n')
                self.model = pickle.load(open('./models/' + self.type + '.sav', 'rb'))
                end_train= timer()
                # print('(', round(timer() - start_train, 4), ')')

                # print('best model: ' , self.model.best_estimator_)
                x,y= test_data[0],test_data[1]
                # print('classifier accuracy (test):', accuracy_score(y, self.model.predict(x)))
                # print('classifier accuracy (train):', accuracy_score(ytrain, self.model.predict(xtrain)))
                # print('train shape: x_mal=',len(np.where(ytrain == 1)[0]), '  x_ben=',len(np.where(ytrain == 0)[0]))

            else:
                # print('TRAINING the ' + self.type + ' CLASSIFIER...')
                self.model.fit(xtrain, ytrain)
                # print('(', round(timer() - start_train,4),')')

                end_train= timer()
                pickle.dump( self.model , open('./models/' + self.type + '.sav', 'wb'))

                # print('best model: ' , self.model.best_estimator_)
                x,y= test_data[0],test_data[1]
                # print('classifier accuracy (test):', accuracy_score(y, self.model.predict(x)))
                # print('classifier accuracy (train):', accuracy_score(ytrain, self.model.predict(xtrain)))
                # print('train shape: x_mal=', len(np.where(ytrain == 1)[0]), '  x_ben=', len(np.where(ytrain == 0)[0]))
            # (fpr, tpr ,treashlod)= roc_curve(ytrain,self.model.predict_proba(xtrain)[:,1])
            # (fpr, tpr ,treashlod)= roc_curve(ytrain,self.model.decision_function(xtrain))

            # roc_auc = auc(fpr, tpr)
            # plt.figure()
            # plt.plot(fpr,tpr)
            # plt.show()

            print('classifier accuracy (train):', accuracy_score(ytrain, self.model.predict(xtrain)))
            print ('classifier confusion_matrix:',confusion_matrix(ytrain,    self.model.predict(xtrain)))
            # print ('classifier roc_auc:',roc_auc)
        return end_train-start_train

def binarize(x):
    return np.where(x > 0.5, np.ones_like(x),np.zeros_like(x))

def train_sarogate_model(target_model , sarogate_model, dataloader, train_data , test_data , batch_size , epochs,device):
    # file_path = './models/'+target_model.type+str(arc)+'surrogate_model.pth'
    file_path = './models/'+target_model.type+'_surrogate_model.pth'
    (x_mal, y_mal), (x_ben, y_ben) = train_data[0], train_data[1]

    if os.path.isfile(file_path):
        # print('\n---------------------')
        # print('LOADING the SURROGATE MODEL WEIGHTS...\n')
        sarogate_model.load_state_dict(torch.load(file_path))
        start_time= timer()

    else:
        # print('\n---------------------')
        # print('TRAINING the Surrogate model querying the targeted CLASSIFIER...\n')

        criterian =  nn.BCELoss()
        loss=[]
        lb = preprocessing.LabelBinarizer()
        lb.fit([0,1])
        # malware_classifier_net.netClassifier = malware_classifier_net(1).to(device)
        optimizer_C = torch.optim.Adam(sarogate_model.parameters(), lr=2e-3 , betas=(0.99, 0.999))
        loss_Classifier_save = 100 #a big possible value
        start_time= timer()
        for epoch in range(epochs):
            # for step in range(x_mal.shape[0] // batch_size):
            #     idx = np.random.randint(0, x_mal.shape[0], batch_size)
            #     xmal_batch = x_mal[idx]
            #     idx = np.random.randint(0, xmal_batch.shape[0], batch_size)
            #     xben_batch = x_ben[idx]
            #     local_batch = np.concatenate([xmal_batch , xben_batch])


            for local_batch, local_lable in dataloader:
                sarogate_model.zero_grad()
                # pred_class = malware_classifier_net(torch.from_numpy(local_batch).float().cuda())
                # pred_class = sarogate_model(torch.from_numpy(local_batch).float().cuda())
                pred_class = sarogate_model(local_batch.float().cuda())
                target_model_label = target_model.model.predict(local_batch)
                # local_lable[(torch.unsqueeze(local_lable, 1)[:, 0] == 0).nonzero()] = [1, 0]
                # local_lable[(torch.unsqueeze(local_lable, 1)[:, 0] == 1).nonzero()] = [0, 1]
                # target_model_label =  np.hstack(( 1 - lb.transform(target_model_label),lb.transform(target_model_label)))
                target_model_label =  np.hstack(( 1 - target_model_label.reshape(-1,1),target_model_label.reshape(-1,1)))
                # local_lable = np.hstack(( 1 - lb.transform(local_lable),lb.transform(local_lable)))
                loss_Classifier =criterian(pred_class,torch.from_numpy(target_model_label).float().cuda() )
                # loss_Classifier =criterian(torch.max(pred_class , 1)[0],local_lable.float().cuda() )

                loss_Classifier.backward()
                optimizer_C.step()
            loss.append(loss_Classifier.cpu().detach().numpy())
            if loss_Classifier <= loss_Classifier_save:
                loss_Classifier_save =loss_Classifier
                torch.save(sarogate_model.state_dict(), file_path)
                # print('model saved')
            # print(epoch , ":",loss_Classifier.cpu().detach().numpy() )
            # torch.save(sarogate_model.state_dict(), file_path)

        # plt.plot(range(len(loss)), loss, label='lr:2e-3, bs=500')
        # plt.show()
    # print('---------------------')
    end_time = timer()
    # print('time:(', round(end_time-start_time,4),')')
    # print("\nACCURACY COMPARISION:")
    # # print('classifier accuracy (train):' ,accuracy_score(np.concatenate([y_mal,y_ben]), binarize(target_model.model.predict(np.concatenate([x_mal,x_ben])))))
    print('classifier accuracy (test):' ,accuracy_score(test_data[1], binarize(target_model.model.predict(test_data[0]))))
    print('sarogate_model accracy:' , accuracy_score(test_data[1],  torch.argmax(sarogate_model(torch.from_numpy(test_data[0]).float().cuda()),1).cpu().detach().numpy()) )
    #
    # # print('\n-----Train data:-----')
    # x_train=np.concatenate([train_data[0][0],train_data[1][0]])
    # y_train=np.concatenate([train_data[0][1],train_data[1][1]])
    # # print('confusion_matrix:')
    # # print ('classifier:',confusion_matrix(y_train,    binarize(target_model.model.predict(x_train))))
    # # print ('sarogate_model :',confusion_matrix(y_train,  torch.argmax(sarogate_model(torch.from_numpy(x_train).float().cuda()),1).cpu().detach().numpy()))
    # # print('-----')
    # # print('roc_auc:')
    # # fpr, tpr,_ = roc_curve(y_train,binarize(target_model.model.predict(x_train)))
    # # train_c_roc_auc = auc(fpr,tpr)
    # # print('classifier:',train_c_roc_auc)
    # fpr, tpr,_ = roc_curve(binarize(target_model.model.predict(x_train)),sarogate_model(torch.from_numpy(x_train).float().cuda())[:,1].cpu().detach().numpy())
    # # fpr, tpr,_ = roc_curve(y_train,sarogate_model(torch.from_numpy(x_train).float().cuda())[:,1].cpu().detach().numpy())
    # train_s_roc_auc = auc(fpr,tpr)
    # print('sarogate_model on train:',train_s_roc_auc)
    #
    # # print('\n-----TEST data:-----')
    #
    # print('confusion_matrix:')
    # print ('classifier:',confusion_matrix(test_data[1],    binarize(target_model.model.predict(test_data[0]))))
    # print ('sarogate_model :',confusion_matrix(test_data[1],  torch.argmax(sarogate_model(torch.from_numpy(test_data[0]).float().cuda()),1).cpu().detach().numpy()))
    # # print('-----')
    # # print('roc_auc:')
    # # fpr, tpr, _ = roc_curve(test_data[1], binarize(target_model.model.predict(test_data[0])))
    # # test_c_roc_auc = auc(fpr, tpr)
    # # print('classifier:', test_c_roc_auc)
    # fpr, tpr,_ = roc_curve(binarize(target_model.model.predict(test_data[0])),sarogate_model(torch.from_numpy(test_data[0]).float().cuda())[:,1].cpu().detach().numpy())
    # # fpr, tpr,_ = roc_curve(test_data[1],sarogate_model(torch.from_numpy(test_data[0]).float().cuda())[:,1].cpu().detach().numpy())
    # test_s_roc_auc = auc(fpr,tpr)
    # print('sarogate_model on test:',test_s_roc_auc)
    # # conf=confusion_matrix(test_data[1], torch.argmax(sarogate_model(torch.from_numpy(test_data[0]).float().cuda()),
    # #                                             1).cpu().detach().numpy())
    # # print('sarogate_model FNR:', conf[1][0]/(conf[1][0]+conf[1][1]))
    # # print('sarogate_model FPR:', conf[0][1]/(conf[0][1]+conf[0][0]))


    return  sarogate_model,end_time-start_time #, (train_s_roc_auc,test_s_roc_auc)

def test_target_model(malware_classifier_net, data):
    x , y =data[0] , data[1]

    x = torch.from_numpy(x).float().cuda()
    predict= malware_classifier_net(x)
    fpr, tpr = roc_curve(y,predict)
    roc_auc = auc(fpr,tpr)
    print('failiure')
    y_pred= torch.argmax(predict,1).cpu().detach().numpy()
    print('roc_auc: ', roc_auc)
    print('accuracy_score: ',accuracy_score(y ,y_pred ))
    print('precision_score: ',precision_score(y ,y_pred ))
    print('recall_score: ',recall_score(y ,y_pred ))
    print('f1_score: ',f1_score(y ,y_pred ))





def train_target_model(targeted_model, dataloader, epochs, device):
    criterian =  nn.BCELoss()
    loss=[]
    lb = preprocessing.LabelBinarizer()
    lb.fit([0,1])
    # malware_classifier_net.netClassifier = malware_classifier_net(1).to(device)
    optimizer_C = torch.optim.Adam(targeted_model.parameters(), lr=2e-3 , betas=(0.99, 0.999))
    for epoch in range(epochs):
        for local_batch, local_lable in dataloader:
            targeted_model.zero_grad()
            # pred_class = malware_classifier_net(torch.from_numpy(local_batch).float().cuda())
            pred_class = targeted_model(local_batch.float().to(device))
            # target_model_label = target_model.model.predict(local_batch)
            # local_lable[(torch.unsqueeze(local_lable, 1)[:, 0] == 0).nonzero()] = [1, 0]
            # local_lable[(torch.unsqueeze(local_lable, 1)[:, 0] == 1).nonzero()] = [0, 1]
            # target_model_label =  np.hstack(( 1 - lb.transform(target_model_label),lb.transform(target_model_label)))
            local_lable = np.hstack(( 1 - lb.transform(local_lable),lb.transform(local_lable)))
            # loss_Classifier =criterian(pred_class,torch.from_numpy(target_model_label).float().cuda() )
            loss_Classifier =criterian(pred_class,torch.from_numpy(local_lable).float().to(device) )
            loss_Classifier.backward()
            optimizer_C.step()
        loss.append(loss_Classifier.cpu().detach().numpy())
        print(epoch , ":",loss_Classifier.cpu().detach().numpy() )
    # torch.save(sarogate_model.state_dict(), file_path)

    plt.plot(range(len(loss)), loss, label='lr:2e-3, bs=500')
    plt.show()

    return  sarogate_model