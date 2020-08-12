import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 input_size,
                 hidden_size,
                 niose_size
                 # image_nc,
                 # box_min,
                 # box_max
                 ):
        # output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        # self.input_nc = image_nc
        # self.output_nc = output_nc
        # self.box_min = box_min
        # self.box_max = box_max

        # self.gen_input_nc = image_nc
        self.netG = models.Mal_Generator(input_size=niose_size, hidden_size=hidden_size, output_size=input_size).to(device)
        self.netDisc = models.Mal_Discriminator(input_size=input_size, hidden_size=hidden_size, output_size=1).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):

            xmal_batch = x[(labels != 0).nonzero()]
            # if (len(xmal_batch.shape) > 2):
            #     xmal_batch = xmal_batch.reshape((xmal_batch.shape[0] * xmal_batch.shape[1]), xmal_batch.shape[2])
            xben_batch = x[(labels == 0).nonzero()]
            # if len(xben_batch.shape) > 2:
            #     xben_batch = xben_batch.reshape((xben_batch.shape[0] * xben_batch.shape[1]), xben_batch.shape[2])
            a=self.netG(xmal_batch)
            # print(a)
            perturbation = torch.where(a>0.5 , torch.ones_like(xmal_batch), torch.zeros_like(xmal_batch))

            # add a clipping trick
            adv_mal = torch.max(xmal_batch,perturbation)
            # adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            # adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(xben_batch)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_mal.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_mal)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb =torch.mean(torch.norm((adv_mal-xmal_batch).view((adv_mal-xmal_batch).shape[0] , -1),1,0))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_mal)
            probs_model = F.softmax(logits_model, dim=1)
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            # real = torch.sum(onehot_labels * probs_model, dim=1)
            # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            # zeros = torch.zeros_like(other)
            # loss_adv = torch.max(real - other, zeros)
            # loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            loss_adv = -F.mse_loss(logits_model, labels)
            # loss_adv = - F.cross_entropy(logits_model, labels.long())

            adv_lambda = 10
            pert_lambda = 1
            #keyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                samples, labels = data
                samples, labels = samples.float().to(self.device), labels.float().to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(samples, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'net_malG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

