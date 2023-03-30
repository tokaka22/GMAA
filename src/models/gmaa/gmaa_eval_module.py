from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image
import torch.nn as nn
import os
import numpy as np

from .crtiterions import LPIPS
from . import pytorch_ssim
from .models.model import Generator_pre, Generator_efgan, Discriminator, Generator
from .FRmodels import irse, ir152, facenet


class AdvLitModule(LightningModule):
    def __init__(
        self,
        config_dict,
        optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.__dict__.update(**config_dict)

        # corresponding to training
        self.LPIPS = LPIPS(net='alex').to(self.device).eval()

        # load au values
        self.conti_au_map = {}
        with open(self.conti_au_path, 'r') as f:
            line = 'begin'
            while line:
                line = f.readline()
                line = line.strip()
                if line not in ['']:
                    conti_au = []

                    split = line.split()
                    values = split[1:]
                    au = []
                    for n in range(self.au_c_dim):
                        au.append(float(values[n])/5.)

                    conti_au.append(np.clip(np.array(au) * self.ratio, 0, 1)) 

                    self.conti_au_map[split[0]] = conti_au

        self.automatic_optimization = False 

        self.axis_ratio = int(self.ori_size / self.image_size)

        self.loss_visualization = {}

        self.terminal_log_path = os.path.join(self.terminal.dir, self.terminal.name)
        self.val_log_path = os.path.join(self.val.dir, self.val.name)

        self.size_dict = {'eye':torch.LongTensor([60, 20]), 'nose':torch.LongTensor([28, 24]), 'mouth':torch.LongTensor([40, 20])}

        self.T_resize_eye = T.Resize([128, 128])
        self.T_tranresize_eye = T.Resize([20, 60])

        self.T_resize_nose = T.Resize([128, 128])
        self.T_tranresize_nose = T.Resize([24, 28])

        self.T_resize_mouth = T.Resize([128, 128])
        self.T_tranresize_mouth = T.Resize([20, 40])

        self.T_resize_detect = T.Resize([128, 128])

        self.pad_dict = {'eye':(34, 34, 54, 54), 'nose':(50, 50, 52, 52), 'mouth':(44, 44, 54, 54)}

        ########## Build G ##########
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num) 

        ########## Build ganimation pre G ##########
        self.PreG = Generator_pre(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        if self.PreG:
            model_save_dir = 'pretrained/exper_edit/full'
            resume_iters = 1
            resume_epoch = 50
            G_path = os.path.join(model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, resume_epoch))
            self.PreG.load_state_dict(torch.load(G_path, map_location='cpu'))
        
        ########## Build ganimation eye G ##########
        self.Pre_eye_G = Generator_efgan(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        if self.Pre_eye_G:
            model_save_dir = 'pretrained/exper_edit/eyes'
            resume_iters = 1
            resume_epoch = 50
            G_path = os.path.join(model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, resume_epoch))
            self.Pre_eye_G.load_state_dict(torch.load(G_path, map_location='cpu'))
        
        ########## Build ganimation nose G ##########
        self.Pre_nose_G = Generator_efgan(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        if self.Pre_nose_G:
            model_save_dir = 'pretrained/exper_edit/nose'
            resume_iters = 1
            resume_epoch = 50
            G_path = os.path.join(model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, resume_epoch))
            self.Pre_nose_G.load_state_dict(torch.load(G_path, map_location='cpu'))

        ########## Build ganimation mouth G ##########
        self.Pre_mouth_G = Generator_efgan(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        if self.Pre_mouth_G:
            model_save_dir = 'pretrained/exper_edit/mouth'
            resume_iters = 1
            resume_epoch = 50
            G_path = os.path.join(model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, resume_epoch))
            self.Pre_mouth_G.load_state_dict(torch.load(G_path, map_location='cpu'))

        ########## Build D ##########
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        ########## Build Target_model ##########
        self.targe_models = {}
        for model in self.attack_model_name_list:  
            if model == 'ir152':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                self.ir152 = ir152.IR_152((112, 112))
                self.ir152.load_state_dict(torch.load('pretrained/FRmodels/ir152.pth'))
                self.targe_models[model].append(self.ir152)
            if model == 'irse50':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                self.irse50 = irse.Backbone(50, 0.6, 'ir_se')
                self.irse50.load_state_dict(torch.load('pretrained/FRmodels/irse50.pth'))
                self.targe_models[model].append(self.irse50)
            if model == 'facenet':
                self.targe_models[model] = []
                self.targe_models[model].append((160, 160))
                self.facenet = facenet.InceptionResnetV1(num_classes=8631)
                self.facenet.load_state_dict(torch.load('pretrained/FRmodels/facenet.pth'))
                self.targe_models[model].append(self.facenet)
            if model == 'mobile_face':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                self.mobile_face = irse.MobileFaceNet(512)
                self.mobile_face.load_state_dict(torch.load('pretrained/FRmodels/mobile_face.pth'))
                self.targe_models[model].append(self.mobile_face)

        self.th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
        self.ASR_class = ['FAR01', 'FAR001', 'FAR0001']

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks

        print(f"output_dir:{self.output_dir}")
        
        pass
    
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
        
    def cos_simi(self, emb_1, emb_2):
        return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

    def cal_adv_loss(self, source, target, model_name, target_models):
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        source_resize = F.interpolate(source, size=input_size, mode='bilinear')
        target_resize = F.interpolate(target, size=input_size, mode='bilinear')
        emb_source = fr_model(source_resize)  
        emb_target = fr_model(target_resize).detach()
        cos_loss = 1 - self.cos_simi(emb_source, emb_target)
        return cos_loss

    def scale_au(self, au_ori):
        max = au_ori.max()
        min = au_ori.min()
        au_scale = (au_ori - min) / (max-min)
        return au_scale * 5
    
    def imFromAttReg(self, att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real
    
    def input_diversity(self, x):  
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False).to(self.device)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0).to(
            self.device)

        return padded if torch.rand(1) < self.diversity_prob else x  

    def input_noise(self, x): 
            rnd = torch.rand(1).to(self.device)
            noise = torch.randn_like(x).to(self.device)
            x_noised = x + rnd * (0.1 ** 0.5) * noise
            x_noised.to(self.device)
            return x_noised if torch.rand(1) < self.diversity_prob else x

    def crop_part(self, x, lm, size):
        half_size = size // 2
        bz = x.shape[0]
        x_crop_list = []

        for b_idx in range(bz):
            crop_start_x = int(lm[b_idx][0] / self.axis_ratio) - half_size[0]
            crop_start_y = int(lm[b_idx][1] / self.axis_ratio) - half_size[1]
            crop_end_x = int(lm[b_idx][0] / self.axis_ratio) + half_size[0]
            crop_end_y = int(lm[b_idx][1] / self.axis_ratio) + half_size[1]

            x_crop = x[b_idx, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]

            x_crop_list.append(x_crop)

        return torch.stack(x_crop_list)

    def crop_detect(self, x, dt):
        bz = x.shape[0]
        x_crop_list = []
        x_shape_list = []

        for b_idx in range(bz):
            crop_start_x = int(dt[b_idx][0] / self.axis_ratio)
            crop_start_y = int(dt[b_idx][1] / self.axis_ratio)
            crop_end_x = int(dt[b_idx][2] / self.axis_ratio)
            crop_end_y = int(dt[b_idx][3] / self.axis_ratio)

            x_crop = x[b_idx, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]

            x_crop_list.append(x_crop)

            h = crop_end_y - crop_start_y
            w = crop_end_x - crop_start_x

            x_shape_list.append((h, w))

        return x_crop_list, x_shape_list

    def resize_detect(self, x_real_detect_crop):
        resize_list = []

        for input in x_real_detect_crop:
            resize_list.append(self.T_resize_detect(input))

        return torch.stack(resize_list)

    def transcrop_detect(self, detect_reses, shapes, dts, fulls): 
        res = fulls.clone()

        bz = detect_reses.shape[0]

        for b_idx in range(bz):
            
            h, w = shapes[b_idx]

            torch_resize = T.Resize([h, w])

            crop_start_x = int(dts[b_idx][0] / self.axis_ratio)
            crop_start_y = int(dts[b_idx][1] / self.axis_ratio)
            crop_end_x = int(dts[b_idx][2] / self.axis_ratio)
            crop_end_y = int(dts[b_idx][3] / self.axis_ratio)

            res[b_idx, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x] = torch_resize(detect_reses[b_idx])

        return res

    def full_loss(self, detect_ganis, shapes, dts, reals, fakes): 

        bz = detect_ganis.shape[0]

        g_loss_detect_list = []
        g_loss_bg_list = []

        for b_idx in range(bz):
            mask = torch.ones(fakes.shape[2:], dtype=torch.bool).to(self.device)
            
            h, w = shapes[b_idx]

            torch_resize = T.Resize([h, w])

            crop_start_x = int(dts[b_idx][0] / self.axis_ratio)
            crop_start_y = int(dts[b_idx][1] / self.axis_ratio)
            crop_end_x = int(dts[b_idx][2] / self.axis_ratio)
            crop_end_y = int(dts[b_idx][3] / self.axis_ratio)

            detect_ganis_resize = torch_resize(detect_ganis[b_idx])

            # detect loss
            if self.g_dt_loss_mode == "perception":
                g_loss_detect = self.LPIPS(fakes[b_idx, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x], detect_ganis_resize).mean() * self.g_LPIPS_dt_w
            elif self.g_dt_loss_mode == "mse":
                perturbation = (fakes[b_idx, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x] - detect_ganis_resize).unsqueeze(dim=0) 
                g_loss_detect = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)) * self.g_p_dt_w 
            elif self.g_dt_loss_mode == "ssim": 
                g_loss_detect = (1 - pytorch_ssim.ssim(fakes[b_idx:b_idx+1, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x], detect_ganis_resize.unsqueeze(dim=0))) * self.g_ssim_dt_w
            else:
                raise RuntimeError("no valid loss mode")

            g_loss_detect_list.append(g_loss_detect)

            # bg loss
            mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = False

            if self.g_bg_loss_mode == "perception":
                g_loss_bg = self.LPIPS(fakes[b_idx,:] * mask, reals[b_idx,:] * mask).mean() * self.g_LPIPS_bg_w
            elif self.g_bg_loss_mode == "mse": 
                perturbation = (fakes[b_idx,:] * mask - reals[b_idx,:] * mask).unsqueeze(dim=0) 
                g_loss_bg = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)) * self.g_p_bg_w
            elif self.g_bg_loss_mode == "ssim": 
                g_loss_bg = (1 - pytorch_ssim.ssim(fakes[b_idx:b_idx+1,:] * mask, reals[b_idx:b_idx+1,:] * mask)) * self.g_ssim_bg_w
            else:
                raise RuntimeError("no valid loss mode")

            g_loss_bg_list.append(g_loss_bg)

        return torch.mean(torch.stack(g_loss_detect_list)), torch.mean(torch.stack(g_loss_bg_list))

    def train_discriminator(self):
        g_opt, d_opt = self.optimizers()

        label_trg = self.label_trg

        # x_fake
        _, x_fake, _ = self.G(self.x_real, label_trg) 

        real_critic, classification_output = self.D(self.x_real) 

        ############## d_loss_critic_real ##############
        d_loss_critic_real = F.mse_loss(real_critic, torch.ones_like(real_critic, device=self.device))
        ############## d_loss_classification ###########
        d_loss_classification = self.lambda_cls * torch.nn.functional.mse_loss(classification_output, self.label_org)
        ############## d_loss_critic_fake ##############
        fake_critic, _ = self.D(x_fake.detach()) 
        d_loss_critic_fake = F.mse_loss(fake_critic, torch.zeros_like(fake_critic, device=self.device))

        # x_hat
        alpha = torch.rand(self.x_real.size(0), 1, 1, 1).to(self.device) 
        x_hat = (alpha * self.x_real.data + (1 - alpha)* x_fake.data).requires_grad_(True) 
        ############## d_loss_gp #######################
        critic_output, _ = self.D(x_hat)
        d_loss_gp = self.lambda_gp * self.gradient_penalty(critic_output, x_hat)
        ############## d_loss_total #####################
        d_loss = d_loss_classification + d_loss_gp + d_loss_critic_real + d_loss_critic_fake

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # Logging.loss画图
        self.loss_visualization['D/loss'] = d_loss.item()
        self.loss_visualization['D/loss_real'] = d_loss_critic_real.item()
        self.loss_visualization['D/loss_fake'] = d_loss_critic_fake.item()
        # self.loss_visualization['D/loss_pur'] = d_loss_critic_purified.item()
        self.loss_visualization['D/loss_cls'] = d_loss_classification.item()
        self.loss_visualization['D/loss_gp'] = d_loss_gp.item()

        self.log("D/loss", d_loss.item(), on_step=True, on_epoch=True)
        self.log("D/loss_real", d_loss_critic_real.item(), on_step=True, on_epoch=True)
        self.log("D/loss_fake", d_loss_critic_fake.item(), on_step=True, on_epoch=True)
        self.log("D/loss_cls", d_loss_classification.item(), on_step=True, on_epoch=True)
        self.log("D/loss_gp", d_loss_gp.item(), on_step=True, on_epoch=True)

    def train_generator(self):
        g_opt, d_opt = self.optimizers()

        label_trg = self.label_trg

        # x_fake
        _, x_fake, _ = self.G(self.x_real, label_trg) 

        critic_output, classification_output = self.D(x_fake)   
        ##################### g_loss_cls ###################
        g_loss_cls = self.lambda_cls * torch.nn.functional.mse_loss(classification_output, label_trg)   
        ##################### g_loss_fake ##################
        g_loss_fake = F.mse_loss(critic_output, torch.ones_like(critic_output, device=self.device)) * self.g_f_w
        ##################### g_loss_perturb ###############
        # x_pre
        with torch.no_grad():
            x_real_detect_crop, x_detect_shape = self.crop_detect(self.x_real, self.dt_org)
            x_real_detect = self.resize_detect(x_real_detect_crop)
            pre_attention_detect, pre_color_detect = self.PreG(x_real_detect, label_trg)
            x_pre_detect = self.imFromAttReg(pre_attention_detect, pre_color_detect, x_real_detect)

        g_loss_detect, g_loss_bg = self.full_loss(x_pre_detect, x_detect_shape, self.dt_org, self.x_real, x_fake)

        # x_pre_eye
        x_real_eye_crop = self.crop_part(self.x_real, self.lm_eye, self.size_dict['eye'])
        x_real_eye = self.T_resize_eye(x_real_eye_crop)
        pre_attention_eye, pre_color_eye = self.Pre_eye_G(x_real_eye, label_trg)
        x_pre_eye = self.imFromAttReg(pre_attention_eye, pre_color_eye, x_real_eye)
        x_pre_eye = self.T_tranresize_eye(x_pre_eye)

        x_fake_eye = self.crop_part(x_fake, self.lm_eye, self.size_dict['eye'])

        if self.g_part_loss_mode == "perception":
            g_loss_eye = self.LPIPS(x_fake_eye, x_pre_eye).mean() * self.g_LPIPS_parts_w
        elif self.g_part_loss_mode == "mse":
            perturbation_eye = x_fake_eye - x_pre_eye 
            g_loss_eye = torch.mean(torch.norm(perturbation_eye.view(perturbation_eye.shape[0], -1), 2, dim=1)) * self.g_p_part_w 
        else:
            raise RuntimeError("no valid loss mode")

        # x_pre_nose
        x_real_nose_crop = self.crop_part(self.x_real, self.lm_nose, self.size_dict['nose'])
        x_real_nose = self.T_resize_nose(x_real_nose_crop)
        pre_attention_nose, pre_color_nose = self.Pre_nose_G(x_real_nose, label_trg)
        x_pre_nose = self.imFromAttReg(pre_attention_nose, pre_color_nose, x_real_nose)
        x_pre_nose = self.T_tranresize_nose(x_pre_nose)

        x_fake_nose = self.crop_part(x_fake, self.lm_nose, self.size_dict['nose'])

        if self.g_part_loss_mode == "perception":
            g_loss_nose = self.LPIPS(x_fake_nose, x_pre_nose).mean() * self.g_LPIPS_parts_w
        elif self.g_part_loss_mode == "mse":
            perturbation_nose = x_fake_nose - x_pre_nose 
            g_loss_nose = torch.mean(torch.norm(perturbation_nose.view(perturbation_nose.shape[0], -1), 2, dim=1)) * self.g_p_part_w 
        else:
            raise RuntimeError("no valid loss mode")

        # x_pre_mouth
        x_real_mouth_crop = self.crop_part(self.x_real, self.lm_mouth, self.size_dict['mouth'])
        x_real_mouth = self.T_resize_mouth(x_real_mouth_crop)
        pre_attention_mouth, pre_color_mouth = self.Pre_mouth_G(x_real_mouth, label_trg)
        x_pre_mouth = self.imFromAttReg(pre_attention_mouth, pre_color_mouth, x_real_mouth)
        x_pre_mouth = self.T_tranresize_mouth(x_pre_mouth)

        x_fake_mouth = self.crop_part(x_fake, self.lm_mouth, self.size_dict['mouth'])

        if self.g_part_loss_mode == "perception":
            g_loss_mouth = self.LPIPS(x_fake_mouth, x_pre_mouth).mean() * self.g_LPIPS_parts_w
        elif self.g_part_loss_mode == "mse":
            perturbation_mouth = x_fake_mouth - x_pre_mouth 
            g_loss_mouth = torch.mean(torch.norm(perturbation_mouth.view(perturbation_mouth.shape[0], -1), 2, dim=1)) * self.g_p_part_w 
        else:
            raise RuntimeError("no valid loss mode")

        ##################### g_loss_adv ##################
        targeted_loss_list = []
        fake_x_diversity = []
        for i in range(self.diversity):  #self.diversity = 5
            fake_x_diversity.append(self.input_diversity(self.input_noise(x_fake)).to(self.device)) 
        for model_name in self.train_model_name_list:
            for i in range(self.diversity):
                target_loss_x = self.cal_adv_loss(fake_x_diversity[i], self.x_trg, model_name, self.targe_models) * self.lambda_adv * 0.5
                targeted_loss_list.append(target_loss_x)
        g_loss_adv = torch.mean(torch.stack(targeted_loss_list)) * self.g_a_w

        ##################### g_loss_total ################
        g_loss = g_loss_fake + g_loss_cls +  g_loss_adv + g_loss_detect + g_loss_bg + g_loss_eye + g_loss_nose + g_loss_mouth
        
        if self.local_step % (self.g_freq * self.g_freq) == 0:
            save_image(torch.cat([self.denorm(self.x_real), self.denorm(x_fake), self.denorm(x_pre_detect)]), self.current_e_train_vis_path +
                   '/{}_0real&fake&pre_.png'.format(self.local_step), nrow=6)
            save_image(torch.cat([F.pad(self.denorm(x_pre_eye), self.pad_dict['eye']), F.pad(self.denorm(x_pre_nose), self.pad_dict['nose']), F.pad(self.denorm(x_pre_mouth), self.pad_dict['mouth'])]), self.current_e_train_vis_path +
                   '/{}_1eye&nose&mouth_.png'.format(self.local_step), nrow=12)
            
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        # Logging
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/adv_loss'] = g_loss_adv.item()
        self.loss_visualization['G/loss_cls'] = g_loss_cls.item()
        self.loss_visualization['G/loss_fake'] = g_loss_fake.item()
        self.loss_visualization['G/loss_dt'] = g_loss_detect.item()
        self.loss_visualization['G/loss_bg'] = g_loss_bg.item()
        self.loss_visualization['G/loss_eye'] = g_loss_eye.item()
        self.loss_visualization['G/loss_nose'] = g_loss_nose.item()
        self.loss_visualization['G/loss_mouth'] = g_loss_mouth.item()

        self.log("G/loss", g_loss.item(), on_step=True, on_epoch=True)
        self.log("G/adv_loss", g_loss_adv.item(), on_step=True, on_epoch=True)
        self.log("G/loss_cls", g_loss_cls.item(), on_step=True, on_epoch=True)
        self.log("G/loss_fake", g_loss_fake.item(), on_step=True, on_epoch=True)
        self.log("G/loss_dt", g_loss_detect.item(), on_step=True, on_epoch=True)
        self.log("G/loss_bg", g_loss_bg.item(), on_step=True, on_epoch=True)
        self.log("G/loss_eye", g_loss_eye.item(), on_step=True, on_epoch=True)
        self.log("G/loss_nose", g_loss_nose.item(), on_step=True, on_epoch=True)
        self.log("G/loss_mouth", g_loss_mouth.item(), on_step=True, on_epoch=True)

    def on_train_epoch_start(self):
        self.current_e_train_vis_path = os.path.join(self.train_vis.dir, self.train_vis.name, str(self.current_epoch))
        os.makedirs(self.current_e_train_vis_path, exist_ok=True)

        for model_name in self.attack_model_name_list:
            fr_model = self.targe_models[model_name][1]
            fr_model.eval()

    def training_step(self, batch: Any, batch_idx: int):

        self.LPIPS = LPIPS(net='alex').to(self.device).eval()
        self.diversity = 5
        self.lambda_adv = 5
        self.diversity_prob = 0.5
        self.resize_rate = 0.9

        self.local_step = batch_idx

        self.x_real, self.x_trg, self.label_org, self.label_trg, self.lm_org, self.dt_org, self.img_idx = batch

        self.lm_eye = self.lm_org[:, 0]
        self.lm_nose = self.lm_org[:, 1]
        self.lm_mouth = self.lm_org[:, 2]

        if self.local_step == 0:
            save_image(self.denorm(self.x_trg), self.current_e_train_vis_path +
            '/trg_.png', nrow=6)

        self.train_discriminator()

        if (self.local_step) % self.g_freq == 0:      
            self.train_generator() 

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return None
    
    def training_step_end(self, batch_outputs):
        log = "[{}/{}], Epoch [{}/{}]".format(self.local_step+1, len(self.trainer.train_dataloader), self.current_epoch, self.trainer.max_epochs)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        
        with open(self.terminal_log_path, 'a') as f:
            f.write(log)
            f.write('\n')

        return None

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass
    
    def cal_ASR(self, source, target, model_name, target_models):
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        source_resize = F.interpolate(source, size=input_size, mode='bilinear')
        target_resize = F.interpolate(target, size=input_size, mode='bilinear')

        emb_source = fr_model(source_resize)  
        emb_target = fr_model(target_resize).detach()

        cos_simi = torch.cosine_similarity(emb_source, emb_target)

        return cos_simi
    
    def on_validation_epoch_start(self):

        self.current_e_val_vis_path = os.path.join(self.val_vis.dir, self.val_vis.name, str(self.current_epoch))
        os.makedirs(self.current_e_val_vis_path, exist_ok=True)

        self.ASR_dict = {}
        for model_name in self.val_model_name_list:
            classes = {}
            for class_name in self.ASR_class:
                classes[class_name] = 0
            self.ASR_dict[model_name] = classes
        
        self.G.train()

    def validation_step(self, batch, batch_idx):
        self.x_real, self.x_trg, self.label_org, self.label_trg, self.lm_org, self.dt_org, self.img_idx = batch

        self.lm_eye = self.lm_org[:, 0]
        self.lm_nose = self.lm_org[:, 1]
        self.lm_mouth = self.lm_org[:, 2]

        _, x_fake, _ = self.G(self.x_real, self.label_trg)


        save_image(torch.cat([self.denorm(self.x_real), self.denorm(x_fake)]), self.current_e_val_vis_path +
                   '/{}_0real&fake&.png'.format(batch_idx))

        for model_name in self.val_model_name_list:
            cos_simi = self.cal_ASR(x_fake, self.x_trg, model_name, self.targe_models)

            if cos_simi.item() > self.th_dict[model_name][0]:
                self.ASR_dict[model_name]['FAR01'] += 1
            if cos_simi.item() > self.th_dict[model_name][1]:
                self.ASR_dict[model_name]['FAR001'] += 1
            if cos_simi.item() > self.th_dict[model_name][2]:
                self.ASR_dict[model_name]['FAR0001'] += 1
        pass
    
    def validation_epoch_end(self, validation_step_outputs):
        total = len(self.trainer.datamodule.data_val)

        if not self.trainer.sanity_checking:
            with open(self.val_log_path, 'a') as f:
                f.write("----------- epoch {} -----------".format(self.current_epoch))
                f.write('\n')
                for model_name in self.val_model_name_list:
                    for class_name in self.ASR_class:
                        log = "{} ASR in {}: {:.4f}".format(model_name, class_name, self.ASR_dict[model_name][class_name] / total)
                        f.write(log)
                        f.write('\n')
        
        pass
    
    def on_test_epoch_start(self):

        self.current_test_vis_root = os.path.join(self.test_vis.dir, self.test_vis.name)
        
        self.G.train()

    def test_step(self, batch: Any, batch_idx: int):

        self.x_real, self.label_org, self.label_trg, self.lm_org, self.dt_org, self.img_idx = batch

        self.current_test_vis_path = os.path.join(self.current_test_vis_root, str(int(self.img_idx.data)))

        os.makedirs(self.current_test_vis_path, exist_ok=True)
        
        for au_idx, key in enumerate(self.conti_au_map):   
            label_trg = self.conti_au_map[key][0]
            label_trg = torch.FloatTensor(label_trg[None,]).to(self.device) 
                
            _, x_fake, _ = self.G(self.x_real, label_trg)

            save_image(self.denorm(x_fake), self.current_test_vis_path +
                   '/{}.png'.format(key.replace('.jpg', '')))

        return 

    def test_epoch_end(self, outputs: List[Any]):


        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        g_opt = torch.optim.Adam(self.G.parameters(), self.hparams.optimizer.g_lr, [self.hparams.optimizer.beta1, self.hparams.optimizer.beta2])
        d_opt = torch.optim.Adam(self.D.parameters(), self.hparams.optimizer.d_lr, [self.hparams.optimizer.beta1, self.hparams.optimizer.beta2])       
        
        return g_opt, d_opt


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
