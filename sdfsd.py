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
    #print('[failiure: %d] [sucsses: %d] [false_negative: %d]' % (failiure, sucsses, false_negative))

    print('>>>>>>>>>>>>>>>>>>>>>>>   attack success rate: ', sucsses / (failiure + sucsses) * 100)
    if len(distrotions) == 0:
        print('average distortion: ', 0)

    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>   average distortion: ', torch.mean(torch.stack(distrotions), dim=0))
def write_results(rows):
    header = ['classifier',	'number of malware samples(original +adersarial)',	'classifier accuracy (train)'	,'classifier accuracy (test)',	'attack success rate(train)','distortion(train)',	'attack success rate(test)'	,	'distortion(test)'	,'failiure' ,'success', 'false negative','time','new_added_adv']
    # header = ['classifier',	'arc', 'train_s_roc_auc', 'test_s_roc_auc']
    with open('malfgsm_restuls.csv','w', newline='') as outfile:
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
    # print(frams.sum(axis=0) / len(list_of_added_features))

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
    # print("CUDA Available: ",torch.cuda.is_available())
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


    classifiers = ['RF']#'RBF_SVM','RF', 'LR', 'MLP']#, 'SVM_R1', 'SVM_R2']#, 'SVM_R3', 'SVM_R4', 'SVM_R5', 'SVM_R6', 'SVM_R7', 'SVM_R8', 'SVM_R9', 'SVM_R10', 'SVM_R11' , 'SVM_R12', 'SVM_R13', 'SVM_R14', 'SVM_R15', 'SVM_R16', 'SVM_R17']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']
    #attackss=['MALJSMA_15','MALJSMA_all','MALFGSM']
    attackss=['MALFGSM']
    featuress=['code']#,'all','code']
    rounds=np.arange(20)

    # architechtures = [10,20,50,100,200]

    excel_rows = []

    attack_time=0
    for classifier in classifiers:
        for attack in attackss:
            for feature in featuress:
                er = []
                avd = []
                print('\n\n========================================================================')
                print('==============================================================')
                print('====================================================')
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
                    # print('\n DATA SHAPE:')
                    print('Train --> [(x_mal:%d, y_mal:%d), (x_ben:%d, y_ben:%d)]' % (
                         x_mal.shape[0], y_mal.shape[0], x_ben.shape[0], y_ben.shape[0]))
                    # print('Test --> [(x:%d, y:%d)]' % (test_data[0].shape[0], test_data[1].shape[0]))
                    ######EXEL######
                    excel_row.append(target)
                    excel_row.append(x_mal.shape[0])

                    print('\n---------------------')
                    print('\n EXPERIMENT ON <<' + target + '>> CLASSIFIER...')
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

                    # print('classifier accuracy (test):', accuracy_score(test_data[1], target_model.model.predict(test_data[0])))
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
                    # print('\n---------------------')
                    # print('ATTACKING BLACK_BOX CLASSIFIER...')
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
                    #####EXEL######
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
                    er.append(np.round(sucsses / (failiure + sucsses) * 100,2))
                    if len(distrotions)>0:
                      avd.append(np.round(torch.mean(torch.stack(distrotions), dim=0).data.cpu().numpy(),2))
                    else:
                      avd.append(0)
                    #####EXEL######
                    excel_row.append(sucsses / (failiure + sucsses) * 100)
                    if len(distrotions) != 0:
                        excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
                    else:
                        excel_row.append(0)

                    del sarogate_model
                    if len(successful_adv_smpl)>2 and len(adversarial_examples)>2:
                        np.save('./MalwareDataset/successful_test_adversarial_samples/'+target+'.npy',successful_adv_smpl)
                    ###########retrain classifier
                        # print('%d unniqu samples out of %d adversrial samples are generated! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
                        adversarial_examples = np.array(list(set(tuple(row_) for row_ in np.unique(np.asarray(adversarial_examples).squeeze(), axis=0))-set(tuple(row) for row in x_mal)))
                        x_mal= np.concatenate([x_mal , adversarial_examples])
                        # print(x_mal.shape[0])
                        excel_row.append(round_time)
                        np.save('./MalwareDataset/adversarial_samples/'+target+'.npy',np.asarray(adversarial_examples).squeeze())
                        # print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
                        excel_row.append(len(adversarial_examples))
                        # print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
                        excel_rows.append(excel_row)
                    else:
                        break
                    if sucsses / (failiure + sucsses) * 100 <1:
                        break
                write_results(excel_rows)
                File_object = open('./output/'+target+'.txt',"w")
                File_object.write(str(excel_row))
                File_object.close()
                # print('\n\n========================================================================')
                # print('==============================================================')
                # print('====================================================')
                print('\n\n---------------------')
                print('er: ', er)
                print('avd:', avd)
                print('====================================================')
                print('==============================================================')
                print('========================================================================')

