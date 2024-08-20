import numpy as np 
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os, wandb, glob, torch
import segmentation_models_pytorch_3branch as smp
from tqdm.auto import tqdm
import albumentations as A
from dataset import SpallingDataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from utils import mIoU


class Trainer:
    def __init__(self, img_path, label_path, save_path, max_lr = 5e-4, epochs = 80, batch = 3, opti_scheme = 'individual', rand = 0, device = 'cuda'):
        self.img_path = img_path
        self.label_path = label_path
        self.save_path = save_path
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch = batch
        self.opti_scheme = opti_scheme
        self.rand = rand
        self.device = device
      
      
    def train_one_epoch(self):
        running_loss = 0
        self.model.train()
        train_iou_score = np.zeros([len(self.train_loader),2])
        prog = tqdm(enumerate(self.train_loader),total=len(self.train_loader))
        for i, data in prog:
            #training phase
            image_tiles, mask_tiles = data
            image = image_tiles.to(self.device); mask = mask_tiles.to(self.device)#; mask_ood = mask_ood_tiles.to(self.device)
            #forward
            output= self.model(image)
            if self.opti_scheme == 'direct':
                loss_3 = self.criterion_3(output[1], mask)
                loss_lov = self.criterion_lov(output[1], mask)
                #loss_ood = self.criterion_ood(output, mask_ood)*5
                loss=(loss_lov + loss_3)/2 
            elif self.opti_scheme == 'individual':
                loss_3_1 = self.criterion_3(output[0], mask)
                loss_lov_1 = self.criterion_lov(output[0], mask)
                loss_3_2 = self.criterion_3(output[1], mask)
                loss_lov_2 = self.criterion_lov(output[1], mask)
                loss_3_3 = self.criterion_3(output[2], mask)
                loss_lov_3 = self.criterion_lov(output[2], mask)
                loss_3_4 = self.criterion_3(output[3], mask)
                loss_lov_4 = self.criterion_lov(output[3], mask)
                loss=(loss_3_1 + loss_lov_1 + loss_3_2 + loss_lov_2 + loss_3_3 + loss_lov_3 + loss_3_4 + loss_lov_4)        
            train_iou_score[i] =  mIoU(output[1], mask)
            #backward
            loss.backward()
            self.optimizer.step() #update weight          
            self.optimizer.zero_grad() #reset gradient
            self.scheduler.step() 
            running_loss += loss.item()
        iou0, iou1=np.nanmean(train_iou_score,axis=0)
        return running_loss/len(self.train_loader), iou0, iou1#, loss_ood.item()
    
    def val_one_epoch(self):
        self.model.eval()
        running_loss = 0
        val_iou_score = np.zeros([len(self.val_loader), 2])
        prog = tqdm(enumerate(self.val_loader),total=len(self.val_loader))
        with torch.no_grad():
            for i, data in prog:
                image_tiles, mask_tiles = data
                image = image_tiles.to(self.device); mask = mask_tiles.to(self.device)#; mask_ood = mask_ood_tiles.to(self.device)
                output= self.model(image)
                if self.opti_scheme == 'direct':
                    loss_3 = self.criterion_3(output[1], mask)
                    loss_lov = self.criterion_lov(output[1], mask)
                    #loss_ood = self.criterion_ood(output, mask_ood)*5
                    loss=(loss_lov + loss_3)/2 
                elif self.opti_scheme == 'individual':
                    loss_3_1 = self.criterion_3(output[0], mask)
                    loss_lov_1 = self.criterion_lov(output[0], mask)
                    loss_3_2 = self.criterion_3(output[1], mask)
                    loss_lov_2 = self.criterion_lov(output[1], mask)
                    loss_3_3 = self.criterion_3(output[2], mask)
                    loss_lov_3 = self.criterion_lov(output[2], mask)
                    loss_3_4 = self.criterion_3(output[3], mask)
                    loss_lov_4 = self.criterion_lov(output[3], mask)
                    loss=(loss_3_1 + loss_lov_1 + loss_3_2 + loss_lov_2 + loss_3_3 + loss_lov_3 + loss_3_4 + loss_lov_4)   
                val_iou_score[i] =  mIoU(output[1], mask)
                running_loss +=  loss.item()
            iou0, iou1=np.nanmean(val_iou_score,axis=0)
        return running_loss/len(self.val_loader), iou0, iou1
    
    def setCEweight(self):
        im_list=glob.glob(os.path.join(self.label_path, '*png'))
        sum_m=0
        total=0
        for im in im_list:
            label = np.array(Image.open(im))/255
            sum_m=sum_m+label.sum()
            total=total+label.shape[0]*label.shape[1]
        return [1.0, total/sum_m]
    
    def fit(self):
        self.train_loader, self.val_loader = self.get_dl()
        self.model = smp.create_model(arch='MAnet',encoder_name='efficientnet-b6', encoder_weights='imagenet'
                                 , classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        weights = self.setCEweight()
        weights = torch.FloatTensor(weights).to(self.device)
        self.criterion_3 = nn.CrossEntropyLoss(weight = weights)
        self.criterion_lov=smp.losses.LovaszLoss(mode='multiclass')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.max_lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, epochs=self.epochs,steps_per_epoch=len(self.train_loader))
        self.model.to(self.device)
        for e in range(self.epochs):
            train_loss, _iou0, train_iou1 = self.train_one_epoch()
            val_loss, _iou0, val_iou1 = self.val_one_epoch()
            wandb.log({"train_loss": train_loss,
                    "train_IoU": train_iou1,
                    
                    "val_loss": val_loss,
                    "val_IoU": val_iou1,
                    
                    })
            self.save_model(e)
    def get_dl(self):
        height=864
        width=864
        t_train = A.Compose([A.Resize(height, width), A.HorizontalFlip(),
            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
            A.GaussNoise(), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])

        t_val = A.Compose([A.Resize(height, width), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        train = open(f'./spalling_data/split/set{self.rand}/train.txt', 'r')
        X_train = [os.path.join(self.img_path, i) for i in train.read().split('\n')[:-1]]
        val = open(f'./spalling_data/split/set{self.rand}/val.txt', 'r')
        X_val = [os.path.join(self.img_path, i) for i in val.read().split('\n')[:-1]]
        test = open(f'./spalling_data/split/set{self.rand}/test.txt', 'r')
        _X_test = [os.path.join(self.img_path, i) for i in test.read().split('\n')[:-1]]
        #datasets
        train_set = SpallingDataset(X_train, t_train)
        val_set = SpallingDataset(X_val, t_val)
        train_loader = DataLoader(train_set, batch_size=3, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=3, shuffle=True, drop_last=True)
        
        return train_loader, val_loader
        
    def save_model(self, e):
        if os.path.isdir(self.save_path)==False:
            os.mkdir(self.save_path)
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{e}.pth"))
        
def main():
    model = Trainer(
        img_path = './spalling_data/image/',
        label_path = './spalling_data/label/',
        save_path = './double_ckpt',
        max_lr = 1e-4,
        epochs = 80,
        batch = 3,
        opti_scheme = 'direct',
        rand = 0,
        device = 'cuda'
    )
    with wandb.init(project="RPL"):
        model.fit()

    
if __name__ == '__main__':
    main()
