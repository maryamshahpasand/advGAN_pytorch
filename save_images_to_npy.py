
import os
import numpy as np
# import imageio
# from PIL import Image
from os import listdir
from os.path import isfile, join
# import SimpleITK as sitk
# # from matplotlib import pyplot
# # a=np.load('C:/Users/45028583/Desktop/splits_v5/npy/Test_FA_1.npy')
# # path= "C:/Users/45028583/Desktop/sonar/MLOs/Augmented_MLO_501by501"
# # Function to rename multiple files
# #
# #     if 'dbimage' in filename:
# #         os.rename(path+'/'+filename, path+'/'+filename[0:7]+"_"+filename[7:])
# mypath = "C:/Users/45028583/Desktop/splits_v6/npy/npy_mlo_nmlo_fa/"
mypath = "C:/Users/45028583/Desktop/splits_v6/npy/file names/"

modes = ['Test' , 'Train']
lables = ['FA', 'MLO' , 'NMLO']
for i in range(1,6):
    for mode in modes:
        for lable in lables:

            print( mode + '_'+lable+'_' + str(i) + '.npy : ')
            xmlo = np.load(mypath + mode + '_'+lable+'_' + str(i) + '.npy')
            print(len(xmlo))

# # mypath = "/home/maryam/sonar/splits_v6/npy/"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
# for filename in os.listdir(mypath):
#     images = []
#     images_101_101=[]
#     images_256_256=[]
#     files = np.load(mypath+filename)
#     for file in files:
#         print(file)
#     # for file in files:
#         # im = imageio.imread(file)
#
#         # itkimage = sitk.ReadImage('F:\\SONAR\\finaldata\\OutputfromAll\\False_alarm_aug\\' + filename)
#         itkimage = sitk.ReadImage(str(file)[2:-1])
#         imgs = sitk.GetArrayFromImage(itkimage)[199:300 , 199:300]
#         # im_frame = Image.open(file)[199:300 , 199:300]
#         # np_frame = np.array(im_frame.getdata())
#         # np_frame_101_101=np_frame.reshape((501,501))[199:300 , 199:300]#.reshape((1,10201))
#         # np_frame_256_256 = np_frame.reshape((501,501))[121:378 , 121:378]#.reshape((1,66049))
#         # images.append(np_frame)
#         # images_101_101.append(np_frame_101_101)
#         # images_256_256.append(np_frame_256_256)
#         # images.append(im_frame)
#         if len(imgs.shape)<3:
#             imgs = np.expand_dims(imgs, axis=2)
#             imgs = np.broadcast_to(imgs, (101,101, 3)).copy()
#             print('I am changing the image')
#         images_101_101.append(imgs)
#         # images_256_256.append(np_frame_256_256)
#     # images=np.array(images)
#     images_101_101=np.array(images_101_101)
#     # images_256_256=np.array(images_256_256)
#     # np.save('/home/maryam/sonar/splits_v6/'+filename,images )
#     # np.shape(images)
#     print([item.shape for item in images_101_101])
#     np.save(mypath+filename[:-4]+"_images_101_101",images_101_101)
#     print(np.load(mypath + filename[:-4] + "_images_101_101.npy").shape)
#     # np.save('/home/maryam/sonar/splits_v6/'+filename[:-4]+"_images_256_256",images_256_256)


# modes = ['Test' , 'Train']
# lables = ['FA', 'MLO' , 'NMLO']
# for mode in modes:
#     # for lable in lables:
#     for i in range(1,6):
#         xmlo=np.load(mypath+mode+'_MLO_'+str(i)+'_images_101_101.npy' )
#         xnmlo=np.load(mypath+mode+'_NMLO_'+str(i)+'_images_101_101.npy')
#         xfa=np.load(mypath+mode+'_FA_'+str(i)+'_images_101_101.npy')
#         x=np.concatenate([xmlo  , xnmlo , xfa])
#         y_mlo = np.hstack ([np.ones([xmlo.shape[0],1])  , np.zeros([xmlo.shape[0],1]) ,  np.zeros([xmlo.shape[0],1])])
#         y_nmlo = np.hstack ([np.zeros([xnmlo.shape[0],1])  , np.ones([xnmlo.shape[0],1]) ,  np.zeros([xnmlo.shape[0],1])])
#         y_fa = np.hstack ([np.zeros([xfa.shape[0],1])  , np.zeros([xfa.shape[0],1]) ,  np.ones([xfa.shape[0],1])])
#         y=np.concatenate([y_mlo  , y_nmlo , y_fa])
#         np.save(mypath+'x_' +mode+'_101_101_'+str(i) ,  x)
#         np.save(mypath+'y_' +mode+'_101_101_'+str(i) ,  y)
#         # x = np.load('/home/maryam/sonar/splits_v6/x_' +mode+'_101_101_'+str(i)+'.npy')
#         # for i in range(len(x)):
#         #     if len(x[i][:][:][:])<3:
#         #         print('error')
#         #         print(np.load('/home/maryam/sonar/splits_v6/x_' +mode+'_101_101_'+str(i)+'.npy').shape)
