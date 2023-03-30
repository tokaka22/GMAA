from builtins import enumerate
import os
from PIL import Image
from tqdm import tqdm
import argparse
import torch
from glob import glob
import logging
import torch.nn.functional as F
from torchvision import transforms as T
from glob import glob
import logging
import cv2
import sys
import re
sys.path.append('src/models/gmaa/FRmodels')
import irse, ir152, facenet


def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im

def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img

def attack_local_models():
    parser = argparse.ArgumentParser()

    parser.add_argument("--FRmodels_pth_path", default="pretrained/FRmodels", help="FRmodel pth path")
    parser.add_argument("--FRmodel_names", type=list, default=['mobile_face'], help="FRmodel for testing")
    parser.add_argument("--num_au", default=6, help="typical_au_interval.txt has 251 aus; typical_au.txt has 6 aus")
    parser.add_argument("--res_root", default="", help="path to generated images during testing") 
    parser.add_argument("--target_path", default="data/CelebA-pairs/id7256/131714.jpg", help="path to the target image")
    parser.add_argument('--device', type=str, default='cuda', help='cuda device')
    
    args = parser.parse_args()

    asr_root = args.res_root.replace('_vis', '_asr')
    asr_eachau_root = os.path.join(asr_root, 'each_au')
    asr_meanau_path = os.path.join(asr_root, 'mean_each_au.txt')

    test_models = {}
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}

    os.makedirs(asr_eachau_root, exist_ok=True)

    try:
        os.remove(asr_meanau_path)
    except OSError:
        pass

    for model_name in args.FRmodel_names:
        if model_name == 'ir152':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load(os.path.join(args.FRmodels_pth_path, 'ir152.pth')))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'irse50':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load(os.path.join(args.FRmodels_pth_path, 'irse50.pth')))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'facenet':
            test_models[model_name] = []
            test_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=args.device)
            fr_model.load_state_dict(torch.load(os.path.join(args.FRmodels_pth_path, 'facenet.pth')))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load(os.path.join(args.FRmodels_pth_path, 'mobile_face.pth')))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)

    au_dict = {}
    for au_idx in range(args.num_au):
        au_dict[str(au_idx)] = 0

    for test_model in test_models.keys():
        size = test_models[test_model][0]
        model = test_models[test_model][1]

        transform = []
        transform.append(T.Resize([128, 128]))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

        target = Image.open(args.target_path)
        target = transform(target).unsqueeze(0).to('cuda')
        target_embbeding = model((F.interpolate(target, size=size, mode='bilinear')))

        for au_idx in au_dict:
            asr_path = os.path.join(asr_eachau_root, f'no.{au_idx}_au.txt')
            os.makedirs(os.path.dirname(asr_path), exist_ok=True)
            try:
                os.remove(asr_path)
            except OSError:
                pass

            FAR01 = 0
            FAR001 = 0
            FAR0001 = 0
            total = 0

            for test_img_name in tqdm(sorted(os.listdir(args.res_root))[:]):
                test_path = os.path.join(args.res_root, test_img_name)
                for img_name in sorted(os.listdir(test_path), key = lambda i:int(re.findall(r'\d+',i)[0]))[:]:
                    if str(int(img_name.split('.')[0])) == au_idx:
                        adv_example = read_img(os.path.join(test_path, img_name), 0.5, 0.5, args.device).to('cuda')
                        ae_embbeding = model((F.interpolate(adv_example, size=size, mode='bilinear')))
                        cos_simi = torch.cosine_similarity(ae_embbeding, target_embbeding)
                        if cos_simi.item() > th_dict[test_model][0]:
                            FAR01 += 1
                        if cos_simi.item() > th_dict[test_model][1]:
                            FAR001 += 1
                        if cos_simi.item() > th_dict[test_model][2]:
                            FAR0001 += 1

                        line = test_img_name + ' ' + "{:.4f}".format(cos_simi.item()) + ' FAR001>thre?: ' + str(cos_simi.item() > th_dict[test_model][1])
                        with open(asr_path, 'a') as f:
                            f.write(line)
                            f.write('\n')
                        total += 1
            
            print(test_model, "ASR in FAR@0.1: {:.4f}, ASR in FAR@0.01: {:.4f}, ASR in FAR@0.001: {:.4f}".
                format(FAR01/total, FAR001/total, FAR0001/total))
            
            with open(asr_path, 'a') as f:
                line = 'all ' + '{:.4f}'.format(FAR001 / total)
                f.write(line)
                f.write('\n')
                line = 'total ' + str(total)
                f.write(line)
                f.write('\n')
            
            au_dict[au_idx] = '{:.4f}'.format(FAR001 / total)
            with open(asr_meanau_path, 'a') as f:
                line = 'au:' + au_idx + ' ' + au_dict[au_idx] + ' total:{:d}'.format(total)
                f.write(line)
                f.write('\n')



if __name__ == '__main__':
    attack_local_models()