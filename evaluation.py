import numpy as np 
import os, glob, torch, cv2, json
import torch.nn as nn
import segmentation_models_pytorch_v2 as smp
import segmentation_models_pytorch_3branch as smp_3b
import albumentations as A
from dataset import SpallingDataset
from albumentations.pytorch import ToTensorV2
from utils import surfd, compute_metrics


class Eval:
    def __init__(self, img_path, label_path, save_path, ckpt, device = 'cuda', rand = 0, baseline = None):
        self.img_path = img_path
        self.label_path = label_path
        self.save_path = save_path
        self.ckpt = ckpt
        self.device = device
        self.rand = rand
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
    
    def eval(self):
        hd95_array = []
        pred_array = []
        label_array = []
        test_set = self.get_ds()
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
            label = test[1]
            pred = self.predict(image)
            hd95_array.append(np.percentile(surfd(pred, label.numpy(), 1, 1), 95))
            pred_array+=list(pred.reshape(-1))
            label_array+=list(label.numpy().reshape(-1))
        hd95 = np.array(hd95_array).mean()
        pred_array = np.array(pred_array).reshape(-1)
        label_array = np.array(label_array).reshape(-1)
        metrics_dict = compute_metrics(pred_array, label_array, background=False)
        metrics_dict['hd95'] = hd95
        json.dump(metrics_dict, open(self.save_path, 'w'))
        
    def get_ds(self):
        t_test = A.Compose([A.Resize(864, 864), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        test = open(f'./spalling_data/split/set{self.rand}/test.txt', 'r')
        X_test = [os.path.join(self.img_path, i) for i in test.read().split('\n')[:-1]]
        test_set = SpallingDataset(X_test, t_test)
        
        return test_set
        
def main():
    for rs in range(7):
        eval = Eval(
                img_path = './spalling_data/image/',
                label_path = './spalling_data/label/',
                save_path = f'./output/performance/performance_MBF_{rs}.json',
                ckpt = '/Data/home/chriswang/project/MBF-UNet/ckpt/MAnet-individual-tileseg(peel)-{'+str(rs)+'}_[1,4,8].pt',
                rand = rs,
                baseline = None
            )
        eval.eval()
        eval = Eval(
            img_path = './spalling_data/image/',
            label_path = './spalling_data/label/',
            save_path = f'./output/performance/performance_MBF_direct_{rs}.json',
            ckpt = '/Data/home/chriswang/project/MBF-UNet/ckpt/MAnet-v2-tileseg(peel)-{'+str(rs)+'}_[1,4,8].pt',
            rand = rs,
            baseline = None
        )
        eval.eval()
    strategies = [
    #'no_aug',
    'no_transfer_aug',
    'no_transfer'
    ]
    for strategy in strategies:
        for rs in range(7):
            eval = Eval(
                    img_path = './spalling_data/image/',
                    label_path = './spalling_data/label/',
                    save_path = f'./output/performance/performance_{strategy}_{rs}.json',
                    ckpt = './ckpt/'+strategy+'-{'+str(rs)+'}.pt',
                    rand = rs,
                    baseline = 'Unet'
                )
            eval.eval()
    archs = [
        'Unet',
        'UnetPlusPlus',
        'MAnet',
        'DeepLabV3Plus'
    ]
    for arch in archs:
        for rs in range(7):
            eval = Eval(
                    img_path = './spalling_data/image/',
                    label_path = './spalling_data/label/',
                    save_path = f'./output/performance/performance_{arch}_{rs}.json',
                    ckpt = './ckpt/MAnet-v2-tileseg(peel)-{'+arch+'}-{'+str(rs)+'}.pt',
                    rand = rs,
                    baseline = arch
                )
            eval.eval()
   
if __name__ == '__main__':
    main()