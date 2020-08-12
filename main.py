import torch
from torch.cuda import device
import attacks
import dataload
from timeit import default_timer as timer
# import sortednp as snp
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset,sampler
from Mal_advGAN import AdvGAN_Attack
# from models import MNIST_target_net, malware_classifeir_net , train_target_model , test_target_model , malware_classifeir , sarogate_model ,train_sarogate_model
from models import  malware_classifeir , surrogate_model ,train_sarogate_model ,malware_classifier_net , train_target_model ,test_target_model
import matplotlib.pyplot as plt
import numpy as np
import collections, functools,operator
import scipy
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
import csv
from os import listdir
from os.path import isfile,join
use_cuda=True
image_nc=1
hidden_size = 200
# niose_size = 100
output_size=1
epochs = 50
batch_size = 128

def print_results(failiure,sucsses,false_negative,title):
    print(title)
    print('number of samples:', failiure + sucsses + false_negative)
    print('[failiure: %d] [sucsses: %d] [false_negative: %d]' % (failiure, sucsses, false_negative))

    print('>>>>>>>>>>>>>>>>>>>>>>>   attack success rate: ', sucsses / (failiure + sucsses) * 100)
    if len(distrotions) == 0:
        print('average distortion: ', 0)

    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>   average distortion: ', torch.mean(torch.stack(distrotions), dim=0))
def write_results(rows):
    header = ['classifier',	'number of malware samples(original +adersarial)',	'classifier accuracy (train)'	,'classifier accuracy (test)',	'attack success rate(train)','distortion(train)',	'attack success rate(test)'	,	'distortion(test)'	,'failiure' ,'success', 'false negative','time','new_added_adv']
    # header = ['classifier',	'arc', 'train_s_roc_auc', 'test_s_roc_auc']
    with open('restuls.csv','w', newline='') as outfile:
        wr = csv.writer(outfile,delimiter=',' , quoting=csv.QUOTE_ALL)
        wr.writerow([h for h in header])
        for i in range(len(rows)):
            wr.writerow(rows[i])
        outfile.close()
def check_added_feature( x, features):
    added_features = []
    for i in range(len(x)):
        unique, counts = np.unique(
            np.where(x.cpu().detach().numpy()[i].astype(int) > 0, features[:, 2], np.zeros(features[:, 2].shape)),
            return_counts=True)
        unique = unique[1:]
        counts = counts[1:]
        added_features.append(dict(zip(unique, counts)))

    batch_added_features = dict(functools.reduce(operator.add, map(collections.Counter, added_features)))
    batch_added_features.update({n: batch_added_features[n] / len(x) for n in batch_added_features.keys()})
    return batch_added_features
def plot_added_featues(list_of_added_features, max_dist,type):
    frams = pd.DataFrame(list_of_added_features)
    print(frams.sum(axis=0) / len(list_of_added_features))

    # frams.to_csv('./models/list_of_added_features_'+str(max_dist)+'_'+type+'.csv')
    # ax = frams.plot.bar(stacked=True)
    # ax.set_xlabel('EPOCH')
    # # for rowNum, row in frams.iterrows():
    # #     ypos = 0
    # #     featuer=0
    # #     for val in row:
    # #         if featuer!=6:
    # #             ypos += val
    # #             ax.text(rowNum, ypos , "{0:.2f}".format(val), color='black' ,ha='center')
    # #         featuer+=1
    # #     ypos = 0
    # plt.title('Average Number of Added Features in Each Epoch')
    # plt.savefig('./added_featues({0}).png'.format(max_dist))
    # print(frams)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train = False
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    #with sarogate _model
    (x_malware,x_benign) , test_data , feature_vectore_size,features = dataload.load_data(train)
    # (c_x_malware,c_x_benign) , c_test_data , c_feature_vectore_size = dataload.load_data(train)
    # (s_x_malware,s_x_benign) , s_test_data , s_feature_vectore_size = dataload.load_data(train)
    x_mal= x_malware
    adv_sample_path = './MalwareDataset/adversarial_samples_malJSMA_all/'

    # in black box setting
    # classifiers = ['SVM_MALJSMA_all', 'SVM_MALJSMA_all_R1', 'SVM_MALJSMA_all_R2', 'SVM_MALJSMA_all_R3', 'SVM_MALJSMA_all_R4', 'SVM_MALJSMA_all_R5', 'SVM_MALJSMA_all_R6', 'SVM_MALJSMA_all_R7', 'SVM_MALJSMA_all_R8', 'SVM_MALJSMA_all_R9', 'SVM_MALJSMA_all_R10', 'SVM_MALJSMA_all_R11' , 'SVM_MALJSMA_all_R12', 'SVM_R13', 'SVM_R14', 'SVM_R15', 'SVM_R16', 'SVM_R17']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']
    # classifiers = ['SVM_MALJSMA_all', 'SVM_MALJSMA_all_R1', 'SVM_MALJSMA_all_R2', 'SVM_MALJSMA_all_R3', 'SVM_MALJSMA_all_R4', 'SVM_MALJSMA_all_R5', 'SVM_MALJSMA_all_R6', 'SVM_MALJSMA_all_R7', 'SVM_MALJSMA_all_R8', 'SVM_MALJSMA_all_R9', 'SVM_MALJSMA_all_R10', 'SVM_MALJSMA_all_R11' , 'SVM_MALJSMA_all_R12', 'SVM_R13', 'SVM_R14', 'SVM_R15', 'SVM_R16', 'SVM_R17']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']


    ####<<<<<<<<generalizability
    # classifiers=[]
    # models_path = './models/SVMmodelsESORICS2020Genralization/'
    # models = [f for f in listdir(models_path) if isfile(join(models_path,f))]
    # for model in models:
    #     classifiers.append(model[0:-4])
    ####generalizability>>>>>>>>>>>


    classifiers = ['MLP','RBF_SVM','RF', 'LR', 'SVM']#, 'SVM_R1', 'SVM_R2']#, 'SVM_R3', 'SVM_R4', 'SVM_R5', 'SVM_R6', 'SVM_R7', 'SVM_R8', 'SVM_R9', 'SVM_R10', 'SVM_R11' , 'SVM_R12', 'SVM_R13', 'SVM_R14', 'SVM_R15', 'SVM_R16', 'SVM_R17']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']
    # attack=['MALJSMA_15','MALJSMA_all','MALFGSM']
    attackss=['MALFGSM']
    featuress=['manifest','code','all']
    rounds=np.arange(40)

    # architechtures = [10,20,50,100,200]

    excel_rows = []

    attack_time=0
    for classifier in classifiers:
        for attack in attackss:
            for feature in featuress:
                er = []
                avd = []
                for round in rounds:
                    target=attack+'_'+classifier+'_'+feature+'_'+str(round)
                    excel_row=[]
                    # adv_samples = [f for f in listdir(adv_sample_path) if isfile(join(adv_sample_path,f))]
                    # for file in adv_samples:
                    #     x_mal=np.concatenate([x_mal,np.load(adv_sample_path+file)])
                    #     print(x_mal.shape[0])
                    x_ben = x_benign[0:len(x_mal)]
                    y_mal = np.ones(x_mal.shape[0])
                    y_ben = np.zeros(x_ben.shape[0])
                    train_data = [(x_mal, y_mal), (x_ben, y_ben)]
                    dataloader = dataload.dataloader(x_malware, x_benign)
                    print('\n DATA SHAPE:')
                    print('Train --> [(x_mal:%d, y_mal:%d), (x_ben:%d, y_ben:%d)]' % (
                        x_mal.shape[0], y_mal.shape[0], x_ben.shape[0], y_ben.shape[0]))
                    print('Test --> [(x:%d, y:%d)]' % (test_data[0].shape[0], test_data[1].shape[0]))
                    ######EXEL######
                    excel_row.append(target)
                    excel_row.append(x_mal.shape[0])
                    print('\n\n========================================================================')
                    print('====================================================')
                    print('====================================')
                    print('EXPERIMENT ON <<' + target + '>> CLASSIFIER:')
                    # for arc in architechtures:
                        ######EXEL######
                        # print('\n====================================')
                        # print('SURROGATE <<'+ str(arc)+ '>> CLASSIFIER:')


                        #train original classifier

                    target_model = malware_classifeir(type=target)
                    train_target_time=target_model.train(train_data,test_data)

                    ######EXEL######
                    excel_row.append(accuracy_score(np.concatenate([y_mal, y_ben]),
                                   target_model.model.predict(np.concatenate([x_mal, x_ben]))))
                    ######EXEL######
                    excel_row.append(accuracy_score(test_data[1],target_model.model.predict(test_data[0])))

                    print('classifier accuracy (test):', accuracy_score(test_data[1], target_model.model.predict(test_data[0])))
                    #use a sarogate model to
                    sarogate_model = surrogate_model(feature_vectore_size, 200, 2).to(device)
                    sarogate_model.eval()
                    # sarogate_model,( train_s_roc_auc,  test_s_roc_auc) = train_sarogate_model(target_model , sarogate_model , dataloader,train_data , test_data, batch_size, epochs, device)
                    sarogate_model, train_sarogate_time  = train_sarogate_model(target_model , sarogate_model , dataloader,train_data , test_data, batch_size, epochs, device)
                    # excel_rows.append([type,arc, train_s_roc_auc*100, test_s_roc_auc*100])

                    round_time = attack_time + train_target_time

                    # FGSM
                    # attack the sarogate_model
                    x_mal_train = train_data[0][0]
                    x_mal_test = test_data[0][np.where(test_data[1] == 1)]
                    distrotions = []
                    failiure = 0
                    sucsses = 0
                    false_negative = 0
                    print('\n---------------------')
                    print('ATTACKING BLACK_BOX CLASSIFIER...')
                    adversarial_examples = []
                    aya = 0
                    xmals, counts = np.unique(x_malware, axis=0, return_counts=True)
                    for i in range(xmals.shape[0]):

                        count=counts[i]
                        xmal=xmals[i]

                        # if i==0:
                        #     print(xmal)
                        if target_model.model.predict(xmal.reshape(1,-1)) == 0:
                            false_negative = false_negative + count
                            # print('this is a flase negative')
                        else:
                            xmal = torch.from_numpy(xmal).float().cuda()
                            result = attacks.fgsm(feature,xmal.unsqueeze(0) ,target_model.model.predict(xmal.unsqueeze(0).cpu().detach().numpy()), sarogate_model, nn.BCELoss(), eps=1)

                            distrotion = torch.sum(result - xmal)
                            if target_model.model.predict(result.cpu().detach().numpy()) == 1:
                                # print('failiure')
                                # print('====================================================================')

                                failiure = failiure + count
                            else:
                                # print('====================================================================')
                                #
                                # print(distrotion)
                                # print('====================================================================')
                                sucsses = sucsses + count
                                for i in range(count):
                                    distrotions.append(distrotion)
                                    adversarial_examples.append(result.cpu().detach().numpy())
                    print_results(failiure,sucsses,false_negative,'<<TRAIN SAMPLES>>')
                    ######EXEL######
                    excel_row.append(sucsses / (failiure + sucsses) * 100)
                    if len(distrotions)!=0:
                        excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
                    else:
                        excel_row.append(0)

                    successful_adv_smpl = []
                    distrotions = []
                    failiure = 0
                    sucsses = 0
                    false_negative = 0
                    added_feature =[]
                    for xmal in x_mal_test:

                        if target_model.model.predict(xmal.reshape(1, -1)) == 0:
                            false_negative = false_negative + 1
                            # print('this is a flase negative')
                        else:
                            xmal = torch.from_numpy(xmal).float().cuda()
                            result = attacks.fgsm(feature,xmal.unsqueeze(0) ,target_model.model.predict(xmal.cpu().detach().numpy().reshape(1,-1)), sarogate_model, nn.BCELoss(), eps=1)
                            distrotion = torch.sum(result - xmal)
                            added_feature.append(check_added_feature(result-xmal , features))

                            if target_model.model.predict(result.cpu().detach().numpy()) == 1:
                                failiure = failiure + 1
                                # print('failiure')
                                # print('====================================================================')

                            else:
                                sucsses = sucsses + 1
                                distrotions.append(distrotion)
                                successful_adv_smpl.append(result)
                                # print('====================================================================')
                                #
                                # print(distrotion)
                                # print('====================================================================')

                    print_results(failiure, sucsses, false_negative, '<<TEST SAMPLES>>')
                    plot_added_featues(added_feature,1,target)
                    er.append(sucsses / (failiure + sucsses) * 100)
                    avd.append(torch.mean(torch.stack(distrotions), dim=0))

                    #####EXEL######
                    excel_row.append(sucsses / (failiure + sucsses) * 100)
                    if len(distrotions) != 0:
                        excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
                    else:
                        excel_row.append(0)

                    del sarogate_model
                    np.save('./MalwareDataset/successful_test_adversarial_samples/'+target+'.npy',successful_adv_smpl)
                    #retrain classifier
                    print('%d unniqu samples out of %d adversrial samples are generated! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
                    adversarial_examples = np.array(list(set(tuple(row_) for row_ in np.unique(np.asarray(adversarial_examples).squeeze(), axis=0))-set(tuple(row) for row in x_mal)))
                    if len(adversarial_examples)>0:
                        x_mal= np.concatenate([x_mal , adversarial_examples])
                        print(x_mal.shape[0])
                        excel_row.append(round_time)
                        np.save('./MalwareDataset/adversarial_samples/'+target+'.npy',np.asarray(adversarial_examples).squeeze())
                        print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
                        excel_row.append(len(adversarial_examples))
                        print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
                        excel_rows.append(excel_row)
                    else:
                        break
                write_results(excel_rows)




    #     # #
    #     #attack the sarogate_model
    #     x_mal_test = test_data[0][np.where(test_data[1] == 1)]
    #     max_distortions =[0.455]
    #     # max_distortions =[1]
    #     for max_dist in max_distortions:
    #         distrotions = []
    #         failiure = 0
    #         sucsses = 0
    #         false_negative=0
    #         print('---------------------')
    #         print('ATTACKING BLACK_BOX CLASSIFIER with max_distortions='+str(max_dist)+'...')
    #         adversarial_examples=[]
    #         aya = 0
    #         attack_start = timer()
    #         # xmals,counts=np.unique(x_mal_train[0:x_malware.shape[0]], axis=0, return_counts=True)
    #         # xmals,counts=np.unique(x_malware, axis=0, return_counts=True)
    #         # for i in range(xmals.shape[0]):
    #         #
    #         #     count=counts[i]
    #         #     xmal=xmals[i]
    #         #
    #         #     # if i==0:
    #         #         # print(xmal)
    #         #     if target_model.model.predict(xmal.reshape(1,-1)) == 0:
    #         #         false_negative = false_negative + count
    #         #         # print('this is a flase negative')
    #         #     else:
    #         #         xmal = torch.from_numpy(xmal).float().cuda()
    #         #         result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=max_dist)
    #         #
    #         #         distrotion = torch.sum(result - xmal)
    #         #         if target_model.model.predict(result.cpu().detach().numpy()) == 1:
    #         #             # print('failiure')
    #         #             # print('====================================================================')
    #         #
    #         #             failiure = failiure + count
    #         #         else:
    #         #             # print('====================================================================')
    #         #             #
    #         #             # print(distrotion)
    #         #             # print('====================================================================')
    #         #             sucsses = sucsses + count
    #         #             for i in range(count):
    #         #                 distrotions.append(distrotion)
    #         #                 adversarial_examples.append(result.cpu().detach().numpy())
    #         # attack_end=timer()
    #         # attack_time=(attack_end-attack_start)+train_sarogate_time
    #         # print_results(failiure,sucsses,false_negative,'<<TRAIN SAMPLES>>')
    #         # ######EXEL######
    #         # excel_row.append(sucsses / (failiure + sucsses) * 100)
    #         # if len(distrotions)!=0:
    #         #     np.save('./models/malJSMA_all/distortion/distortion_'+type+'.npy',distrotions)
    #         #     excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
    #         # else:
    #         #     excel_row.append(0)
    #         successful_adv_smpl =[]
    #         distrotions = []
    #         failiure = 0
    #         sucsses = 0
    #         false_negative = 0
    #         added_feature=[]
    #         # xmals, counts = np.unique(x_mal_test, axis=0, return_counts=True)
    #         xmals, counts = np.unique(x_mal_test, axis=0, return_counts=True)
    #         for i in range(xmals.shape[0]):
    #             count=counts[i]
    #             xmal=xmals[i]
    #             if target_model.model.predict(xmal.reshape(1,-1)) == 0:
    #                 false_negative = false_negative + count
    #                 # print('this is a flase negative')
    #             else:
    #                 xmal = torch.from_numpy(xmal).float().cuda()
    #                 result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=max_dist)
    #                 distrotion = torch.sum(result - xmal)
    #                 if target_model.model.predict(result.cpu().detach().numpy()) == 1:
    #                     failiure = failiure + count
    #                     # print('failiure')
    #                     # print('====================================================================')
    #
    #                 else:
    #                     sucsses = sucsses + count
    #                     for i in range(count):
    #                         distrotions.append(distrotion)
    #                         successful_adv_smpl.append(result)
    #                         added_feature.append(check_added_feature(result-xmal , features))
    #                     # print('====================================================================')
    #                     #
    #                     # print(distrotion)
    #                     # print('====================================================================')
    #
    #         print_results(failiure,sucsses,false_negative , '<<TEST SAMPLES>>')
    #         plot_added_featues(added_feature,max_dist,type)
    #         #####EXEL######
    #         # excel_row.append(sucsses / (failiure + sucsses) * 100)
    #     #     if len(distrotions)!=0:
    #     #         excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
    #     #     else:
    #     #         excel_row.append(0)
    #     #     excel_row.append(failiure)
    #     #     excel_row.append(sucsses)
    #     #     excel_row.append(false_negative)
    #     #
    #     #
    #     #     del sarogate_model
    #     #     np.save('./MalwareDataset/successful_test_adversarial_samples/'+type+'.npy',successful_adv_smpl)
    #     #     #retrain classifier
    #     #     print('%d unniqu samples out of %d adversrial samples are generated! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
    #     #     adversarial_examples = np.array(list(set(tuple(row_) for row_ in np.unique(np.asarray(adversarial_examples).squeeze(), axis=0))-set(tuple(row) for row in x_mal)))
    #     #     x_mal= np.concatenate([x_mal , adversarial_examples])
    #     #     # print(x_mal.shape[0])
    #     #     excel_row.append(round_time)
    #     #     np.save('./MalwareDataset/adversarial_samples_malJSMA_all/'+type+'.npy',np.asarray(adversarial_examples).squeeze())
    #     #     print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
    #     #     excel_row.append(len(adversarial_examples))
    #     #     print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
    #     #
    #     #     excel_rows.append(excel_row)
    #     #
    #     # write_results(excel_rows)
    # # #
    # # #
    # # #
    # # #
    # # #
    # # #
    # # #
    # # #
    # # #
    # # #
    # # # # dataloader, feature_vectore_size = dataload.load_data(train)
    # # # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # # # targeted_model.eval()
    # # # # pretrained_model = "./models/malware_classifier_net.pth"
    # # # #
    # # # # if train:
    # # # #     train_target_model(targeted_model, dataloader, epochs, device)
    # # # #     torch.save(targeted_model.state_dict(), pretrained_model)
    # # # #     test_target_model(targeted_model, test_data)
    # # # #
    # # # # else:
    # # # #     targeted_model.load_state_dict(torch.load(pretrained_model))
    # # # #     test_target_model(targeted_model, test_data)
    # # # #
    # # # # x_mal = test_data[0][np.where(test_data[1] == 1)]
    # # # # y_mal = np.ones(x_mal.shape[0])
    # # # # distrotions = []
    # # # # failiure = 0
    # # # # sucsses = 0
    # # # # for xmal in x_mal:
    # # # #
    # # # #     xmal=torch.from_numpy(xmal).float().cuda()
    # # # #     if torch.argmax(targeted_model(xmal.unsqueeze(0))) == 0:
    # # # #         print('this is a flase negative')
    # # # #     else:
    # # # #         result = attacks.jsma(targeted_model, xmal.unsqueeze(0), 0, max_distortion=0.1)
    # # # #         distrotion = torch.sum(result - xmal)
    # # # #         if torch.argmax(targeted_model(result)) == 1:
    # # # #             # print('failiure')
    # # # #             failiure = failiure + 1
    # # # #         else:
    # # # #             # print(distrotion)
    # # # #             # print('====================================================================')
    # # # #             sucsses = sucsses + 1
    # # # # distrotions.append(distrotion)
    # # # # print('number of samples:', failiure + sucsses)
    # # # # print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
    # # # # print('changes: ', torch.mean(torch.stack(distrotions)))
    # # #
    # # #
    # # # # pretrained_model = "./MNIST_target_model.pth"
    # # # # pretrained_model = "./malware_classifier_net.pth"
    # # # # train_target_model(targeted_model, dataloader, epochs, device)
    # # #
    # # # # targeted_model.load_state_dict(torch.load(pretrained_model))
    # # #
    # # #
    # # # # targeted_model = MNIST_target_net().to(device)
    # # # # # plt.figure()
    # # # # for l_r in [1e-3, 1e-4, 3e-4, 1e-5, 5e-5]:
    # # # #     targeted_model = malware_classifier_net(feature_vectore_size , 200 , 2).to(device)
    # # # #     loss=train_target_model(targeted_model, dataloader, epochs, device , l_r)
    # # # #     plt.plot(range(len(loss)), loss, label='lr:' + str(l_r))
    # # # #
    # # # # plt.xlabel('Epoch')
    # # # # plt.ylabel('Loss')
    # # # # plt.title('Loss curve for different learning rate')
    # # # # plt.legend()
    # # # # plt.show()
    # # # # batch_size = [1000 , 3000, 5000 , 7000]
    # # # # losses=[]
    # # # # for bs in batch_size:
    # # # #     targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # # #
    # # # #     class_sample_count = np.array(
    # # # #     [len(np.where(y == t)[0]) for t in np.unique(y)])
    # # # #     weight = 1. / class_sample_count
    # # # #     samples_weight = []
    # # # #     for t in range(len(y) - 1):
    # # # #         samples_weight.append(weight[int(y[t])])
    # # # #     Sampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    # # # #     data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    # # # #     data_loader = DataLoader(data, batch_size=bs, sampler=Sampler , drop_last = True)
    # # # #     losses.append(train_target_model(targeted_model, data_loader, epochs, device))
    # # # #
    # # # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # # # criterian = nn.BCELoss()
    # # # # loss = []
    # # # # # malware_classifier_net.netClassifier = malware_classifier_net(1).to(device)
    # # # # optimizer_C = torch.optim.Adam(targeted_model.parameters(), lr=2e-3, betas=(0.9, 0.999))
    # # # # for epoch in range(epochs):
    # # # #     targeted_model.zero_grad()
    # # # #     # pred_class = malware_classifier_net(torch.from_numpy(local_batch).float().cuda())
    # # # #     pred_class = targeted_model(torch.from_numpy(x).float().cuda())
    # # # #     loss_Classifier = criterian(pred_class[:, 0],torch.from_numpy(y).float().float().cuda())
    # # # #     # loss_Classifier =criterian(torch.max(pred_class , 1)[0],local_lable.float().cuda() )
    # # # #     loss_Classifier.backward()
    # # # #     optimizer_C.step()
    # # # #     loss.append(loss_Classifier.cpu().detach().numpy())
    # # # #     print(epoch, ":", loss_Classifier.cpu().detach().numpy())
    # # # # losses.append(loss)
    # # # # plt.figure()
    # # # # plt.plot(range(len(losses[0])), losses[0] , label='bs:1000')
    # # # # plt.plot(range(len(losses[1])), losses[1] , label='bs:3000')
    # # # # plt.plot(range(len(losses[2])), losses[2] , label='bs:5000')
    # # # # plt.plot(range(len(losses[3])), losses[3] , label='bs:7000')
    # # # # plt.plot(range(len(losses[4])), losses[4] , label='bs:no data loader')
    # # # #
    # # # #
    # # # # plt.xlabel('Epoch')
    # # # # plt.ylabel('Loss')
    # # # # plt.title('Loss curve when using dataloader for balancd batch')
    # # # # plt.legend()
    # # # # plt.show()
    # # #
    # # #
    # # #
    # # # # MNIST train dataset and dataloader declaration
    # # # # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # # # # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # # # # distrotions=[]
    # # # # failiure=0
    # # # # sucsses =0
    # # # # x = scipy.sparse.load_npz('./MalwareDataset/x_test.npz').toarray()
    # # # # y = np.load('./MalwareDataset/y_test.npy')
    # # # #
    # # # # x_mal = x[np.where(y == 1)]
    # # # # x_ben = x[np.where(y == 0)]
    # # # # x_ben = x_ben[0:len(x_mal)]
    # # # # y_mal = np.ones(x_mal.shape[0])
    # # # # y_ben = np.zeros(x_ben.shape[0])
    # # # # x = np.concatenate([x_mal, x_ben])
    # # # # y = np.concatenate([y_mal, y_ben])
    # # # # feature_vectore_size = x_mal.shape[1]
    # # # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # # # targeted_model.load_state_dict(torch.load(pretrained_model))
    # # # # targeted_model.eval()
    # # # # model_num_labels = 1
    # # # # # x= torch.from_numpy(x).float().cuda()
    # # # # # predict= targeted_model(x)
    # # # # # print('failiure')
    # # # # # y_pred= torch.argmax(predict,1).cpu().detach().numpy()
    # # # # # print('accuracy_score: ',accuracy_score(y ,y_pred ))
    # # # # for xmal in x_mal:
    # # # #     xmal=torch.from_numpy(xmal).float().cuda()
    # # # #     result = attacks.jsma(targeted_model, xmal, 0, max_distortion=0.5)
    # # # #     distrotion = torch.sum(result - xmal)
    # # # #     if torch.argmax(targeted_model(result)) == 1:
    # # # #         print('failiure')
    # # # #         failiure = failiure + 1
    # # # #     else:
    # # # #         print('sucssesssssssssssssssssssssssssssssssss')
    # # # #         print(distrotion)
    # # # #         sucsses = sucsses + 1
    # # # # distrotions.append(distrotion)
    # # # # print('number of samples:', failiure + sucsses)
    # # # # print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
    # # # # print('changes: ', torch.mean(torch.stack(distrotions)))
    # # #
    # # # # for i, data in enumerate(dataloader, start=0):
    # # # #     malware_samples = data[0][(data[1] == 1).nonzero()].squeeze()
    # # # #     # result = attacks.jsma(malware_samples.float().to(device), data[1][(data[1] == 1).nonzero()], targeted_model, nn.BCELoss(), 1)
    # # # #     #
    # # # #     #
    # # # #     # result = attacks.fgsm(malware_samples.float().to(device), data[1][(data[1] == 1).nonzero()], targeted_model, nn.BCELoss(), 1)
    # # # #
    # # # #     # samples, labels = data
    # # # #     malware_samples = malware_samples.float().to(device)
    # # # #
    # # # #     for sample in malware_samples:
    # # # #
    # # # #         result=attacks.jsma(targeted_model, sample, 0, max_distortion=0.5)
    # # # #         # result=attacks.fgsm(sample,0 , targeted_model,nn.BCELoss(),1)
    # # # #         # inputs, targets, model, criterion, eps
    # # # #         distrotion = torch.sum(result-sample)
    # # # #         if torch.argmax(targeted_model(result))==1:
    # # # #             # print('failiure')
    # # # #             failiure=failiure+ 1
    # # # #         else:
    # # # #             # print('sucsses')
    # # # #             sucsses=sucsses+1
    # # # #     distrotions.append(distrotion)
    # # # # print('number of samples:' , failiure+sucsses)
    # # # # print('attack success rate: ' , sucsses/(failiure+sucsses) * 100)
    # # # # print('changes: ' , torch.mean(torch.stack(distrotions)))
    # # # # # advGAN = AdvGAN_Attack(device,
    # # # # #                           targeted_model,
    # # # # #                           model_num_labels,
    # # # # #                           feature_vectore_size,
    # # # # #                           hidden_size,
    # # # # #                           feature_vectore_size
    # # # # #                           # image_nc,
    # # # #                           # BOX_MIN,
    # # # #                           # BOX_MAX
    # # # #                        )
    # # # #
    # # # # advGAN.train(dataloader, epochs)
