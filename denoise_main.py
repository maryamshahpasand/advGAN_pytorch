import torch
from torch.cuda import device
import attacks
import dataload
import torch.optim as optim
from torch.utils.data import DataLoader ,Dataset
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset,sampler
from Mal_advGAN import AdvGAN_Attack
# from models import MNIST_target_net, malware_classifeir_net , train_target_model , test_target_model , malware_classifeir , sarogate_model ,train_sarogate_model
from models import  malware_classifeir , surrogate_model ,train_sarogate_model ,malware_classifier_net , train_target_model ,test_target_model
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
import csv

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

    print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
    if len(distrotions) == 0:
        print('average distortion: ', 0)

    else:
        print('average distortion: ', torch.mean(torch.stack(distrotions), dim=0))
def write_results(rows):
    header = ['classifier',	'number of malware samples(original +adersarial)',	'classifier accuracy (train)'	,'classifier accuracy (test)',	'attack success rate(train)','distortion(train)',	'attack success rate(test)'	,	'distortion(test)'	,'failiure' ,'success', 'false negative']
    with open('restuls.csv','w', newline='') as outfile:
        wr = csv.writer(outfile,delimiter=',' , quoting=csv.QUOTE_ALL)
        wr.writerow([h for h in header])
        for i in range(len(rows)):
            wr.writerow(rows[i])
        outfile.close()
class noisedDataset(Dataset):

    def __init__(self, datasetnoised, datasetclean, labels, transform):
        self.dirty = datasetnoised
        self.clean = datasetclean
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dirty)

    def __getitem__(self, idx):
        xDirty = self.dirty[idx]
        xClean = self.clean[idx]
        y = self.labels[idx]

        # if self.transform != None:
        #     xNoise = self.transform(xDirty)
        #     xClean = self.transform(xClean)

        return (xDirty, xClean, y)
class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(542, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid()

        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 542),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    train = False
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    #with sarogate _model

    x_malware = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/xtrain_mal.npz' ).toarray()
    x_benign = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/xtrain_ben.npz' ).toarray()
    xtest_mal = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/xtest_mal.npz' ).toarray()
    xtest_ben = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/xtest_ben.npz' ).toarray()

    y_malware = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/ytrain_mal.npz' ).toarray()
    y_benign = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/ytrain_ben.npz' ).toarray()
    ytest_mal = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/ytest_mal.npz' ).toarray()
    ytest_ben = scipy.sparse.load_npz('./MalwareDataset/Drebin_with_families/ytest_ben.npz' ).toarray()

    x_test = np.concatenate([xtest_mal, xtest_ben])
    y_test = np.reshape(np.concatenate([ytest_mal, ytest_ben], axis=1 ),-1)
    test_data = (x_test,y_test)
    feature_vectore_size = x_malware.shape[1]

    x_mal = x_malware
    # in black box setting
    classifiers = [ 'SVM']#, 'SVM_R1', 'SVM_R2', 'SVM_R3', 'SVM_R4', 'SVM_R5', 'SVM_R6', 'SVM_R7', 'SVM_R8', 'SVM_R9', 'SVM_R10', 'SVM_R11' , 'SVM_R12', 'SVM_R13', 'SVM_R14', 'SVM_R15']# ,'RF' , 'RBF_SVM', 'LR', 'DT' , 'KNN', 'MLP']# ,'DNN']
    excel_rows=[]
    for type in classifiers:
        ######EXEL######
        excel_row=[type]
        print('\n====================================')
        print('EXPERIMENT ON <<'+ type+ '>> CLASSIFIER:')

        x_ben = x_benign[0:len(x_mal)]
        y_mal = np.ones(x_mal.shape[0])
        y_ben = np.zeros(x_ben.shape[0])
        train_data=[(x_mal, y_mal), (x_ben, y_ben)]
        dataloader = dataload.dataloader(x_mal,x_ben)
        print('\n DATA SHAPE:')
        print('Train --> [(x_mal:%d, y_mal:%d), (x_ben:%d, y_ben:%d)]' %(x_mal.shape[0],y_mal.shape[0],x_ben.shape[0],y_ben.shape[0]))
        print('Test --> [(x:%d, y:%d)]' %(x_test.shape[0],y_test.shape[0]))
        ######EXEL######
        excel_row.append(x_mal.shape[0])

        #train original classifier

        target_model = malware_classifeir(type=type)
        target_model.train(train_data)

        # ######EXEL######
        # excel_row.append(accuracy_score(np.concatenate([y_mal, y_ben]),
        #                target_model.model.predict(np.concatenate([x_mal, x_ben]))))
        # ######EXEL######
        # excel_row.append(accuracy_score(test_data[1],target_model.model.predict(test_data[0])))

        #use a sarogate model to mimic the classifier
        sarogate_model = surrogate_model(feature_vectore_size, 200, 2).to(device)
        sarogate_model.eval()
        sarogate_model = train_sarogate_model(target_model , sarogate_model , dataloader,train_data , test_data, batch_size, epochs, device)
    # FGSM
    #     # attack the sarogate_model
    #     x_mal_train = train_data[0][0]
    #     x_mal_test = test_data[0][np.where(test_data[1] == 1)]
    #     distrotions = []
    #     failiure = 0
    #     sucsses = 0
    #     false_negative = 0
    #     print('---------------------')
    #     print('ATTACKING BLACK_BOX CLASSIFIER...')
    #     adversarial_examples = []
    #     aya = 0
    #     for xmal in x_mal_train[0:x_malware.shape[0]]:
    #         if target_model.model.predict(xmal.reshape(1,-1)) == 0:
    #             false_negative = false_negative + 1
    #             # print('this is a flase negative')
    #         else:
    #             xmal = torch.from_numpy(xmal).float().cuda()
    #             result = attacks.fgsm(xmal.unsqueeze(0) ,target_model.model.predict(xmal.unsqueeze(0).cpu().detach().numpy()), sarogate_model, nn.BCELoss(), eps=1)
    #
    #             distrotion = torch.sum(result - xmal)
    #             if target_model.model.predict(result.cpu().detach().numpy()) == 1:
    #                 # print('failiure')
    #                 # print('====================================================================')
    #
    #                 failiure = failiure + 1
    #             else:
    #                 # print('====================================================================')
    #                 #
    #                 # print(distrotion)
    #                 # print('====================================================================')
    #                 sucsses = sucsses + 1
    #                 distrotions.append(distrotion)
    #                 adversarial_examples.append(result.cpu().detach().numpy())
    #     print_results(failiure,sucsses,false_negative,'<<TRAIN SAMPLES>>')
    #     ######EXEL######
    #     excel_row.append(sucsses / (failiure + sucsses) * 100)
    #     if len(distrotions)!=0:
    #         excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
    #     else:
    #         excel_row.append(0)
    #
    #     successful_adv_smpl = []
    #     distrotions = []
    #     failiure = 0
    #     sucsses = 0
    #     false_negative = 0
    #
    #     for xmal in x_mal_test:
    #
    #         if target_model.model.predict(xmal.reshape(1, -1)) == 0:
    #             false_negative = false_negative + 1
    #             # print('this is a flase negative')
    #         else:
    #             xmal = torch.from_numpy(xmal).float().cuda()
    #             result = attacks.fgsm(xmal.unsqueeze(0) ,target_model.model.predict(result.cpu().detach().numpy()), sarogate_model, nn.BCELoss(), eps=1)
    #             distrotion = torch.sum(result - xmal)
    #             if target_model.model.predict(result.cpu().detach().numpy()) == 1:
    #                 failiure = failiure + 1
    #                 # print('failiure')
    #                 # print('====================================================================')
    #
    #             else:
    #                 sucsses = sucsses + 1
    #                 distrotions.append(distrotion)
    #                 successful_adv_smpl.append(result)
    #                 # print('====================================================================')
    #                 #
    #                 # print(distrotion)
    #                 # print('====================================================================')
    #
    #     print_results(failiure, sucsses, false_negative, '<<TEST SAMPLES>>')
    #
    #     #####EXEL######
    #     excel_row.append(sucsses / (failiure + sucsses) * 100)
    #     if len(distrotions) != 0:
    #         excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
    #     else:
    #         excel_row.append(0)
    #
    #     del sarogate_model
    #     np.save('./MalwareDataset/adversarial_samples/successful_adv_smpl_against_'+type+'.npy',successful_adv_smpl)
    #     #retrain classifier
    #     if len(np.unique(np.asarray(adversarial_examples).squeeze(), axis=0))!=0:
    #         print('%d unniqu samples out of %d adversrial samples are generated and are added to training set! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
    #         adversarial_examples = np.unique(np.asarray(adversarial_examples).squeeze(), axis=0)
    #         x_mal= np.concatenate([x_mal , adversarial_examples])
    #
    #
    #     # print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
    #     print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
    #     #
    #     # excel_rows.append(excel_row)
    #     # write_results(excel_rows)



        #attack the sarogate_model
        x_mal_train = train_data[0][0]
        x_mal_test = test_data[0][np.where(test_data[1] == 1)]
        x_ben_test = test_data[0][np.where(test_data[1] == 0)]



        distrotions = []
        failiure = 0
        sucsses = 0
        false_negative=0
        print('---------------------')
        print('ATTACKING BLACK_BOX CLASSIFIER...')
        adversarial_examples=[]
        clean_examples=[]
        aya = 0
        for xmal in x_mal_train[0:x_malware.shape[0]]:
        # for xmal in x_mal_train[0:20]:
            if target_model.model.predict(xmal.reshape(1,-1)) == 0:
                false_negative = false_negative + 1
                # print('this is a flase negative')
            else:
                xmal = torch.from_numpy(xmal).float().cuda()
                result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=0.027)

                distrotion = torch.sum(result - xmal)
                if target_model.model.predict(result.cpu().detach().numpy()) == 1:
                    # print('failiure')
                    # print('====================================================================')

                    failiure = failiure + 1
                else:
                    # print('====================================================================')
                    #
                    # print(distrotion)
                    # print('====================================================================')
                    sucsses = sucsses + 1
                    distrotions.append(distrotion)
                    #save clean examples
                    clean_examples.append(xmal.cpu().detach().numpy())
                    adversarial_examples.append(result.view(-1).cpu().detach().numpy())
        print_results(failiure,sucsses,false_negative,'<<TRAIN SAMPLES>>')
        ######EXEL######
        excel_row.append(sucsses / (failiure + sucsses) * 100)
        if len(distrotions)!=0:
            excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
        else:
            excel_row.append(0)
        np.save('./MalwareDataset/adversarial_samples/successful_adv_train_smpl_against_'+type+'.npy',adversarial_examples)
        np.save('./MalwareDataset/adversarial_samples/clean_train_smpl_used_against_'+type+'.npy',clean_examples)
        # adversarial_examples= np.load('./MalwareDataset/adversarial_samples/successful_adv_train_smpl_against_'+type+'.npy')
        # clean_examples = np.load('./MalwareDataset/adversarial_samples/clean_train_smpl_used_against_'+type+'.npy')



        #create clean and noised datasets for training autoendcoder
        ytrain =np.ones(np.array(clean_examples).shape[0])
        tsfms = transforms.Compose([transforms.ToTensor()])
        trainset = noisedDataset(np.array(adversarial_examples), np.array(clean_examples), ytrain, tsfms)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)


        ################### train autoencoder ###################

        model = denoising_model().to(device)
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000002 , betas=(0.99, 0.999))

        epochs = 100
        l = len(trainloader)
        losslist = list()
        epochloss = 0
        running_loss = 0

        successful_adv_smpl =[]
        distrotions = []
        failiure = 0
        sucsses = 0
        deniose_sucsses =0
        false_negative = 0

        for epoch in range(epochs):

            print("Entering Epoch: ", epoch)
            for dirty, clean, label in tqdm((trainloader)):
                dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
                clean = clean.view(clean.size(0), -1).type(torch.FloatTensor)
                dirty, clean = dirty.to(device), clean.to(device)

                # -----------------Forward Pass----------------------
                output = model(dirty)
                loss = criterion(output, clean)
                # -----------------Backward Pass---------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epochloss += loss.item()
            # -----------------Log-------------------------------
            losslist.append(running_loss / l)
            running_loss = 0
            print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))
        file_path = './models/denoising_AE_model.pth'
        torch.save(sarogate_model.state_dict(), file_path)
        plt.plot(range(len(losslist)), losslist)
        plt.show()

        ####################################################
        ################### test samples ###################
        ####################################################

        denised_mal_examples=[]
        adversarial_examples=[]
        clean_examples=[]
        denoised_distortions=[]
        distrotions =[]
        file_path = './models/denoising_AE_model.pth'
        model.load_state_dict(torch.load(file_path), strict=False)
        for xmal in x_mal_test:

            if target_model.model.predict(xmal.reshape(1,-1)) == 0:
                false_negative = false_negative + 1
                # print('this is a flase negative')
            else:
                xmal = torch.from_numpy(xmal).float().cuda()
                result = attacks.jsma(target_model, sarogate_model, xmal.unsqueeze(0), 0, max_distortion=0.027)
                distrotion = torch.sum(result - xmal)
                if target_model.model.predict(result.cpu().detach().numpy()) == 1:
                    failiure = failiure + 1
                    print('failiure')
                    print('====================================================================')

                else:
                    print('success')
                    print('====================================================================')
                    print('distrotion:', distrotion)
                    sucsses = sucsses + 1
                    distrotions.append(distrotion)
                    clean_examples.append(xmal.cpu().detach().numpy())
                    adversarial_examples.append(result.view(-1).cpu().detach().numpy())
                    output = model(result)
                    output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu().detach().numpy()
                    denised_mal_examples.append(output)
                    if target_model.model.predict(output) ==1 :
                        denoised_distortion= torch.abs(torch.sum(torch.from_numpy(output).float().cuda() - xmal))
                        denoised_distortions.append(denoised_distortion)
                        print('denoised distrotion:', denoised_distortion)
                        deniose_sucsses = deniose_sucsses + 1
                    # successful_adv_smpl.append(result)
                    # print('====================================================================')
                    #
                    # print(distrotion)
                    # print('====================================================================')

        print_results(failiure,sucsses,false_negative , '<<TEST SAMPLES>>')
        #####EXEL######
        excel_row.append(sucsses / (failiure + sucsses) * 100)
        if len(distrotions)!=0:
            excel_row.append(torch.mean(torch.stack(distrotions), dim=0))
        else:
            excel_row.append(0)


        del sarogate_model
        np.save('./MalwareDataset/adversarial_samples/successful_adv_test_smpl_against_'+type+'.npy',adversarial_examples)
        np.save('./MalwareDataset/adversarial_samples/clean_test_smpl_used_against_'+type+'.npy',clean_examples)
        np.save('./MalwareDataset/adversarial_samples/denoised_mal_test_smpl_used_against_'+type+'.npy',denised_mal_examples)

        #retrain classifier
        print('%d unniqu samples out of %d adversrial samples are generated and are added to training set! '%(len(np.unique(np.asarray(adversarial_examples).squeeze() , axis=0)), len(adversarial_examples)))
        print ('****successful denoised:',deniose_sucsses ,'successful denoised portion:', deniose_sucsses/sucsses)
        print ('****deleted distortion :',torch.mean(denoised_distortion))

        ####################################################
        ################### test samples ###################
        ####################################################


        x_ben_test = test_data[0][np.where(test_data[1] == 0)]
        x_mal_test = test_data[0][np.where(test_data[1] == 1)]

        output = model(torch.from_numpy(x_ben_test).float().cuda())
        output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu().detach().numpy()
        VAE_samples_lables= target_model.model.predict(output)
        print('conf matrix for ben test:',confusion_matrix(np.zeros(output.shape[0]), VAE_samples_lables))

        output = model(torch.from_numpy(x_mal_test).float().cuda())
        output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu().detach().numpy()
        VAE_samples_lables= target_model.model.predict(output.cpu().detach().numpy())

        print('conf matrix for mal test:',confusion_matrix(np.ones(output.shape[0]), VAE_samples_lables))


        # adversarial_examples = np.unique(np.asarray(adversarial_examples).squeeze(), axis=0)
        # x_mal= np.concatenate([x_mal , adversarial_examples])
        #
        #
        # # print('%d new adversarial samples are added to training set!' % len(adversarial_examples))
        # print('%d uniqu malwar samples out of %d samples are in training set!' %(len(np.unique(x_mal , axis=0)), len(x_mal)))
        #
        # excel_rows.append(excel_row)
    write_results(excel_rows)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # dataloader, feature_vectore_size = dataload.load_data(train)
    # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # targeted_model.eval()
    # # pretrained_model = "./models/malware_classifier_net.pth"
    # #
    # # if train:
    # #     train_target_model(targeted_model, dataloader, epochs, device)
    # #     torch.save(targeted_model.state_dict(), pretrained_model)
    # #     test_target_model(targeted_model, test_data)
    # #
    # # else:
    # #     targeted_model.load_state_dict(torch.load(pretrained_model))
    # #     test_target_model(targeted_model, test_data)
    # #
    # # x_mal = test_data[0][np.where(test_data[1] == 1)]
    # # y_mal = np.ones(x_mal.shape[0])
    # # distrotions = []
    # # failiure = 0
    # # sucsses = 0
    # # for xmal in x_mal:
    # #
    # #     xmal=torch.from_numpy(xmal).float().cuda()
    # #     if torch.argmax(targeted_model(xmal.unsqueeze(0))) == 0:
    # #         print('this is a flase negative')
    # #     else:
    # #         result = attacks.jsma(targeted_model, xmal.unsqueeze(0), 0, max_distortion=0.1)
    # #         distrotion = torch.sum(result - xmal)
    # #         if torch.argmax(targeted_model(result)) == 1:
    # #             # print('failiure')
    # #             failiure = failiure + 1
    # #         else:
    # #             # print(distrotion)
    # #             # print('====================================================================')
    # #             sucsses = sucsses + 1
    # # distrotions.append(distrotion)
    # # print('number of samples:', failiure + sucsses)
    # # print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
    # # print('changes: ', torch.mean(torch.stack(distrotions)))
    #
    #
    # # pretrained_model = "./MNIST_target_model.pth"
    # # pretrained_model = "./malware_classifier_net.pth"
    # # train_target_model(targeted_model, dataloader, epochs, device)
    #
    # # targeted_model.load_state_dict(torch.load(pretrained_model))
    #
    #
    # # targeted_model = MNIST_target_net().to(device)
    # # # plt.figure()
    # # for l_r in [1e-3, 1e-4, 3e-4, 1e-5, 5e-5]:
    # #     targeted_model = malware_classifier_net(feature_vectore_size , 200 , 2).to(device)
    # #     loss=train_target_model(targeted_model, dataloader, epochs, device , l_r)
    # #     plt.plot(range(len(loss)), loss, label='lr:' + str(l_r))
    # #
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Loss')
    # # plt.title('Loss curve for different learning rate')
    # # plt.legend()
    # # plt.show()
    # # batch_size = [1000 , 3000, 5000 , 7000]
    # # losses=[]
    # # for bs in batch_size:
    # #     targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # #
    # #     class_sample_count = np.array(
    # #     [len(np.where(y == t)[0]) for t in np.unique(y)])
    # #     weight = 1. / class_sample_count
    # #     samples_weight = []
    # #     for t in range(len(y) - 1):
    # #         samples_weight.append(weight[int(y[t])])
    # #     Sampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    # #     data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    # #     data_loader = DataLoader(data, batch_size=bs, sampler=Sampler , drop_last = True)
    # #     losses.append(train_target_model(targeted_model, data_loader, epochs, device))
    # #
    # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # criterian = nn.BCELoss()
    # # loss = []
    # # # malware_classifier_net.netClassifier = malware_classifier_net(1).to(device)
    # # optimizer_C = torch.optim.Adam(targeted_model.parameters(), lr=2e-3, betas=(0.9, 0.999))
    # # for epoch in range(epochs):
    # #     targeted_model.zero_grad()
    # #     # pred_class = malware_classifier_net(torch.from_numpy(local_batch).float().cuda())
    # #     pred_class = targeted_model(torch.from_numpy(x).float().cuda())
    # #     loss_Classifier = criterian(pred_class[:, 0],torch.from_numpy(y).float().float().cuda())
    # #     # loss_Classifier =criterian(torch.max(pred_class , 1)[0],local_lable.float().cuda() )
    # #     loss_Classifier.backward()
    # #     optimizer_C.step()
    # #     loss.append(loss_Classifier.cpu().detach().numpy())
    # #     print(epoch, ":", loss_Classifier.cpu().detach().numpy())
    # # losses.append(loss)
    # # plt.figure()
    # # plt.plot(range(len(losses[0])), losses[0] , label='bs:1000')
    # # plt.plot(range(len(losses[1])), losses[1] , label='bs:3000')
    # # plt.plot(range(len(losses[2])), losses[2] , label='bs:5000')
    # # plt.plot(range(len(losses[3])), losses[3] , label='bs:7000')
    # # plt.plot(range(len(losses[4])), losses[4] , label='bs:no data loader')
    # #
    # #
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Loss')
    # # plt.title('Loss curve when using dataloader for balancd batch')
    # # plt.legend()
    # # plt.show()
    #
    #
    #
    # # MNIST train dataset and dataloader declaration
    # # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # # distrotions=[]
    # # failiure=0
    # # sucsses =0
    # # x = scipy.sparse.load_npz('./MalwareDataset/x_test.npz').toarray()
    # # y = np.load('./MalwareDataset/y_test.npy')
    # #
    # # x_mal = x[np.where(y == 1)]
    # # x_ben = x[np.where(y == 0)]
    # # x_ben = x_ben[0:len(x_mal)]
    # # y_mal = np.ones(x_mal.shape[0])
    # # y_ben = np.zeros(x_ben.shape[0])
    # # x = np.concatenate([x_mal, x_ben])
    # # y = np.concatenate([y_mal, y_ben])
    # # feature_vectore_size = x_mal.shape[1]
    # # targeted_model = malware_classifier_net(feature_vectore_size, 200, 2).to(device)
    # # targeted_model.load_state_dict(torch.load(pretrained_model))
    # # targeted_model.eval()
    # # model_num_labels = 1
    # # # x= torch.from_numpy(x).float().cuda()
    # # # predict= targeted_model(x)
    # # # print('failiure')
    # # # y_pred= torch.argmax(predict,1).cpu().detach().numpy()
    # # # print('accuracy_score: ',accuracy_score(y ,y_pred ))
    # # for xmal in x_mal:
    # #     xmal=torch.from_numpy(xmal).float().cuda()
    # #     result = attacks.jsma(targeted_model, xmal, 0, max_distortion=0.5)
    # #     distrotion = torch.sum(result - xmal)
    # #     if torch.argmax(targeted_model(result)) == 1:
    # #         print('failiure')
    # #         failiure = failiure + 1
    # #     else:
    # #         print('sucssesssssssssssssssssssssssssssssssss')
    # #         print(distrotion)
    # #         sucsses = sucsses + 1
    # # distrotions.append(distrotion)
    # # print('number of samples:', failiure + sucsses)
    # # print('attack success rate: ', sucsses / (failiure + sucsses) * 100)
    # # print('changes: ', torch.mean(torch.stack(distrotions)))
    #
    # # for i, data in enumerate(dataloader, start=0):
    # #     malware_samples = data[0][(data[1] == 1).nonzero()].squeeze()
    # #     # result = attacks.jsma(malware_samples.float().to(device), data[1][(data[1] == 1).nonzero()], targeted_model, nn.BCELoss(), 1)
    # #     #
    # #     #
    # #     # result = attacks.fgsm(malware_samples.float().to(device), data[1][(data[1] == 1).nonzero()], targeted_model, nn.BCELoss(), 1)
    # #
    # #     # samples, labels = data
    # #     malware_samples = malware_samples.float().to(device)
    # #
    # #     for sample in malware_samples:
    # #
    # #         result=attacks.jsma(targeted_model, sample, 0, max_distortion=0.5)
    # #         # result=attacks.fgsm(sample,0 , targeted_model,nn.BCELoss(),1)
    # #         # inputs, targets, model, criterion, eps
    # #         distrotion = torch.sum(result-sample)
    # #         if torch.argmax(targeted_model(result))==1:
    # #             # print('failiure')
    # #             failiure=failiure+ 1
    # #         else:
    # #             # print('sucsses')
    # #             sucsses=sucsses+1
    # #     distrotions.append(distrotion)
    # # print('number of samples:' , failiure+sucsses)
    # # print('attack success rate: ' , sucsses/(failiure+sucsses) * 100)
    # # print('changes: ' , torch.mean(torch.stack(distrotions)))
    # # # advGAN = AdvGAN_Attack(device,
    # # #                           targeted_model,
    # # #                           model_num_labels,
    # # #                           feature_vectore_size,
    # # #                           hidden_size,
    # # #                           feature_vectore_size
    # # #                           # image_nc,
    # #                           # BOX_MIN,
    # #                           # BOX_MAX
    # #                        )
    # #
    # # advGAN.train(dataloader, epochs)
