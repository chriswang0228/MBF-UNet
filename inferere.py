import numpy as np 
import pandas as pd
import os, glob, torch, cv2
import torch.nn as nn
import segmentation_models_pytorch_v2 as smp
import segmentation_models_pytorch_3branch as smp_3b
import albumentations as A
from dataset import SpallingDataset
from albumentations.pytorch import ToTensorV2


class Infer:
    def __init__(self, img_path, label_path, save_path, ckpt, rand=0, baseline = None, device = 'cuda'):
        self.img_path = img_path
        self.label_path = label_path
        self.save_path = save_path
        self.ckpt = ckpt
        self.rand = rand
        self.device = device
        self.baseline = baseline
    
    def predict(self, image):
        image = image.unsqueeze(0)
        h, w = image.shape[-2:]
        #image = nn.functional.interpolate(image,
        #            size=[864, 864], # (height, width)
        #            mode='bilinear',
        #            align_corners=False)
        image=image.to(self.device)
        with torch.no_grad():
            if self.baseline is not None:
                output = self.model(image)
            else:
                output = self.model(image)[1]
            output = torch.argmax(output, dim=1)
            output = output.cpu().squeeze(0).numpy()
            #output = cv2.resize(output, [w, h], interpolation=cv2.INTER_NEAREST)
        return output
    
    def infer(self):
        X_test, test_set = self.get_ds()
        if self.baseline is not None:
            if self.baseline=='DeepLabV3Plus':
                self.model = smp.create_model(arch='DeepLabV3Plus',encoder_name='efficientnet-b6', encoder_weights='imagenet'
                                              , classes=2, activation=None, encoder_depth=5).to(self.device)
            else:
                self.model = smp.create_model(arch=self.baseline,encoder_name='efficientnet-b6', encoder_weights='imagenet'
                                    , classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        else:
            self.model = smp_3b.create_model(arch='MAnet',encoder_name='efficientnet-b6', encoder_weights='imagenet'
                                    , classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model.eval()
        for i,test in enumerate(test_set):
            image = test[0]
            pred = self.predict(image)
            cv2.imwrite(os.path.join(self.save_path, X_test[i].split('/')[-1]), pred*255)
        
    def get_ds(self):
        t_test = A.Compose([A.Resize(864, 864), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        test = open(f'./spalling_data/split/set{self.rand}/test.txt', 'r')
        X_test = [os.path.join(self.img_path, i) for i in test.read().split('\n')[:-1]]
        test_set = SpallingDataset(X_test, t_test)
        
        return X_test, test_set
        
def main():
    infer = Infer(
        img_path = './spalling_data/image/',
        label_path = './spalling_data/label/',
        save_path = './output/prediction/',
        ckpt = './ckpt/tileseg_ckpt.pt',
        rand = 0
    )
    infer.infer()
   
if __name__ == '__main__':
    main()