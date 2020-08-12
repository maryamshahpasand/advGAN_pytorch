import torch
# from torch.cuda import device
# import attacks
# import dataload
# import torchvision.datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader,TensorDataset,sampler
# from Mal_advGAN import AdvGAN_Attack
# # from models import MNIST_target_net, malware_classifeir_net , train_target_model , test_target_model , malware_classifeir , sarogate_model ,train_sarogate_model
# from models import  malware_classifeir , surrogate_model ,train_sarogate_model ,malware_classifier_net , train_target_model ,test_target_model
# import matplotlib.pyplot as plt
import numpy as np
# import scipy
# import torch.nn as nn
# from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
# import csv
import os.path
import pickle

classifiers = [ 'MALJSAM_12_SVM_R10', 'MALJSAM_52_SVM_R10','MALFGSM_mnfst_SVM_R7','MALGAN_mnfst_SVM_R10','MALGAN_all_SVM_R10','opt_MALGAN_mnfst_SVM_R10']
adv_sapmles = ['MALJSAM_15', 'MALJSAM_52' ,'MALFGSM_mnfst','MALGAN_mnfst','MALGAN_all','opt_MALGAN_mnfst' ]
excel_rows=[]
for classifier in classifiers:
    for adv_sample in adv_sapmles:
        successful_adv_smples = []
        successful_adv_smpl = np.load('./MalwareDataset/generalizability/'+adv_sample+'.npy')
        if successful_adv_smpl[0].dtype is torch.float32:
            for i in range(0 ,len(successful_adv_smpl)-1):
                # print(i)
                successful_adv_smples.append(successful_adv_smpl[i].cpu().detach().numpy().squeeze())
            successful_adv_smpl=successful_adv_smples
        # if os.path.isfile('./models/generalizability/' + classifier + '.sav'):
        model = pickle.load(open('./models/generalizability/' + classifier + '.sav', 'rb'))
        y_predict = model.predict(successful_adv_smpl)
        attack_success_rate = (len(y_predict)-sum(y_predict))/len(y_predict)
        print('attack_success_rate of adv samples from <<' + adv_sample +'>> against <<'+ classifier + '>> :' ,attack_success_rate)


#
#
# use_cuda=True
# image_nc=1
# hidden_size = 200
# # niose_size = 100
# output_size=1
# epochs = 50
# batch_size = 128
#
# def print_results(failiure,sucsses,false_negative,title):
#     print(title)
#     print('number of samples:', failiure + sucsses + false_negative)
#     print('[failiure: %d] [sucsses: %d] [false_negative: %d]' % (failiure, sucsses, false_negative))
#
#     print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
#     if len(distrotions) == 0:
#         print('average distortion: ', 0)
#
#     else:
#         print('average distortion: ', torch.mean(torch.stack(distrotions), dim=0))
# def write_results(rows):
#     header = ['classifier',	'number of malware samples(original +adersarial)',	'classifier accuracy (train)'	,'classifier accuracy (test)',	'attack success rate(train)','distortion(train)',	'attack success rate(test)'	,	'distortion(test)'	,'failiure' ,'success', 'false negative']
#     with open('restuls.csv','w', newline='') as outfile:
#         wr = csv.writer(outfile,delimiter=',' , quoting=csv.QUOTE_ALL)
#         wr.writerow([h for h in header])
#         for i in range(len(rows)):
#             wr.writerow(rows[i])
#         outfile.close()
#
# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()
#
#     train = False
#     # Define what device we are using
#     print("CUDA Available: ",torch.cuda.is_available())
#     device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
#     #with sarogate _model
#
#     (x_malware,x_benign) , test_data , feature_vectore_size = dataload.load_data(train)
#     x_mal= x_malware
#
#     # in black box setting
#     classifiers = ['RF']#'SVM' , 'SVM_R1', 'SVM_R2', 'SVM_R3', 'SVM_R4', 'SVM_R5', 'SVM_R6', 'SVM_R7', 'SVM_R8', 'SVM_R9', 'SVM_R10']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']
#     excel_rows=[]
#     for type in classifiers:
#         ######EXEL######
#         excel_row=[type]
#         print('\n====================================')
#         print('EXPERIMENT ON <<'+ type+ '>> CLASSIFIER:')
#
#         x_ben = x_benign[0:len(x_mal)]
#         y_mal = np.ones(x_mal.shape[0])
#         y_ben = np.zeros(x_ben.shape[0])
#         train_data=[(x_mal, y_mal), (x_ben, y_ben)]
#         dataloader = dataload.dataloader(x_mal,x_ben)
#         print('\n DATA SHAPE:')
#         print('Train --> [(x_mal:%d, y_mal:%d), (x_ben:%d, y_ben:%d)]' %(x_mal.shape[0],y_mal.shape[0],x_ben.shape[0],y_ben.shape[0]))
#         print('Test --> [(x:%d, y:%d)]' %(test_data[0].shape[0],test_data[1].shape[0]))
#         ######EXEL######
#         excel_row.append(x_mal.shape[0])
#
#         #train original classifier
#
#         target_model = malware_classifeir(type=type)
#         target_model.train(train_data)
#
#         ######EXEL######
#         excel_row.append(accuracy_score(np.concatenate([y_mal, y_ben]),
#                        target_model.model.predict(np.concatenate([x_mal, x_ben]))))
#         ######EXEL######
#         excel_row.append(accuracy_score(test_data[1],target_model.model.predict(test_data[0])))
#
#         #use a sarogate model to
#         sarogate_model = surrogate_model(feature_vectore_size, 200, 2).to(device)
#         sarogate_model.eval()
#         sarogate_model = train_sarogate_model(target_model , sarogate_model , dataloader,train_data , test_data, batch_size, epochs, device)
#
#
#
#         #attack the sarogate_model
#         x_mal_train = train_data[0][0]
#         x_mal_test = test_data[0][np.where(test_data[1] == 1)]
#
#         distrotions = []
#         failiure = 0
#         sucsses = 0
#         false_negative=0
#         print('---------------------')
#         print('ATTACKING BLACK_BOX CLASSIFIER...')
#         adversarial_examples=[]
#         aya = 0
#         for xmal in x_mal_train[0:x_malware.shape[0]]:
#             if target_model.model.predict(xmal.reshape(1,-1)) == 0:
#                 false_negative = false_negative + 1
#                 # print('this is a flase negative')
#             else:
#                 xmal = torch.from_numpy(xmal).float().cuda()
#                 result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=1)
#
#                 distrotion = torch.sum(result - xmal)
#                 if target_model.model.predict(result.cpu().detach().numpy()) == 1:
#                     # print('failiure')
#                     # print('====================================================================')
#
#                     failiure = failiure + 1
#                 else:
#                     # print('====================================================================')
#                     #
#                     # print(distrotion)
#                     # print('====================================================================')
#                     sucsses = sucsses + 1
#                     distrotions.append(distrotion)
#                     adversarial_examples.append(result.cpu().detach().numpy())
#         print_results(failiure,sucsses,false_negative,'<<TRAIN SAMPLES>>')
#         ######EXEL######
#         excel_row.append(sucsses / (failiure + sucsses) * 100)
#         if len(distrotions)!=0:
#             excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
#         else:
#             excel_row.append(0)
#
#         successful_adv_smpl =[]
#         distrotions = []
#         failiure = 0
#         sucsses = 0
#         false_negative = 0
#
#         for xmal in x_mal_test:
#
#             if target_model.model.predict(xmal.reshape(1,-1)) == 0:
#                 false_negative = false_negative + 1
#                 # print('this is a flase negative')
#             else:
#                 xmal = torch.from_numpy(xmal).float().cuda()
#                 result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=1)
#                 distrotion = torch.sum(result - xmal)
#                 if target_model.model.predict(result.cpu().detach().numpy()) == 1:
#                     failiure = failiure + 1
#                 else:
#                     sucsses = sucsses + 1
#                     distrotions.append(distrotion)
#                     successful_adv_smpl.append(result)
#
#         print_results(failiure,sucsses,false_negative , '<<TEST SAMPLES>>')
#
#         ######EXEL######
#         excel_row.append(sucsses / (failiure + sucsses) * 100)
#         if len(distrotions)!=0:
#             excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
#         else:
#             excel_row.append(0)
#
#
#         del sarogate_model
#         np.save('./MalwareDataset/adversarial_samples/successful_adv_smpl_against_'+type+'.npy',successful_adv_smpl)
#         #retrain classifier
#         print('%d unniqu samples out of %d adversrial samples are generated and are added to training set! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
#         adversarial_examples = np.unique(np.asarray(adversarial_examples).squeeze(), axis=0)
#         x_mal= np.concatenate([x_mal , adversarial_examples])
#
#         # print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
#         print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
#
#         excel_rows.append(excel_row)
#     write_results(excel_rows)
#
#
#
#
#
#
