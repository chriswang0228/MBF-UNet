import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SpallingDataset(Dataset):
    
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_path = self.X[idx].replace('/image', '/label')
        mask = np.array(Image.open(mask_path[:-4]+'_label.png'))/255

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].long()


        if self.transform is None:
            img = Image.fromarray(img)
                    
        return img, mask