import os
import time
import requests
from json import JSONDecoder
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import re

def attack_faceplusplus():
    parser = argparse.ArgumentParser()

    parser.add_argument("--res_root", default="", help="path to generated images during testing")
    parser.add_argument("--num_au", default=6, help="typical_au_interval.txt has 251 aus; typical_au.txt has 6 aus")
    parser.add_argument("--target_path", default="data/CelebA-pairs/id7256/131714.jpg", help="path to the target image")

    args = parser.parse_args()

    faceplusplus_root = args.res_root.replace('_vis', '_faceplusplus')
    faceplusplus_eachau_root = os.path.join(faceplusplus_root, 'each_au')
    faceplusplus_meanau_path = os.path.join(faceplusplus_root, 'mean_each_au.txt')

    os.makedirs(faceplusplus_eachau_root, exist_ok=True)


    try:
        os.remove(faceplusplus_meanau_path)
    except OSError:
        pass
    
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
    key = ""  # your own api key here
    secret = ""  # your own api secret here
    data = {"api_key": key, "api_secret": secret}
    target_name = os.path.abspath(args.target_path)

    au_dict = {}
    for au_idx in range(args.num_au):
        au_dict[str(au_idx)] = 0

    for au_idx in au_dict: 
        confidence = 0
        total = 0      
        faceplusplus_path = os.path.join(faceplusplus_eachau_root, f'no.{au_idx}_au.txt')
        os.makedirs(os.path.dirname(faceplusplus_path), exist_ok=True)
        try:
            os.remove(faceplusplus_path)
        except OSError:
            pass

        for test_img_name in tqdm(sorted(os.listdir(args.res_root))[:]):
            test_path = os.path.join(args.res_root, test_img_name)
            for img_name in sorted(os.listdir(test_path), key = lambda i:int(re.findall(r'\d+',i)[0]))[:]:

                if str(int(img_name.split('.')[0])) == au_idx:
                    source_name = os.path.join(test_path, img_name)
                    files = {"image_file1": open(source_name, "rb"), "image_file2": open(target_name, "rb")}
                    time.sleep(0.1)
                    response = requests.post(http_url, data=data, files=files)
                    req_con = response.content.decode('utf-8')
                    req_dict = JSONDecoder().decode(req_con)
                    
                    if 'confidence' in req_dict.keys():
                        confidence += req_dict['confidence']
                        line = test_img_name + ' ' + "{:.4f}".format(req_dict['confidence'])
                        with open(faceplusplus_path, 'a') as f:
                            f.write(line)
                            f.write('\n')
                        total += 1
                    else:
                        print('failed')
                        continue

        print('Face++ confidence: {:.4f}'.format(confidence / total))

        with open(faceplusplus_path, 'a') as f:
            line = 'all ' + '{:.4f}'.format(confidence / total)
            f.write(line)
            f.write('\n')
            line = 'total ' + str(total)
            f.write(line)
            f.write('\n')
        
        au_dict[au_idx] = '{:.4f}'.format(confidence / total)
        with open(faceplusplus_meanau_path, 'a') as f:
            line = 'au:' + au_idx + ' ' + au_dict[au_idx] + ' total:{:d}'.format(total)
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    attack_faceplusplus()

