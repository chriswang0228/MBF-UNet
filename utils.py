import torch
import numpy as np
from scipy.ndimage import morphology
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def mIoU(pred_mask, mask, smooth=1e-10):
    with torch.no_grad():
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, 2): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
    return iou_per_class

def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(bool))
    input_2 = np.atleast_1d(input2.astype(bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    
    return sds

def compute_metrics(pred_mask, mask, background=False):
    iou_per_class=[]
    acc_per_class=[]
    precision_per_class=[]
    recall_per_class=[]
    f1_per_class=[]
    if background:
        target = np.unique(mask)
    else:
        target = np.unique(mask)[1:]
    for clas in target: #loop per pixel class
        true_class = pred_mask == clas
        true_label = mask == clas
        intersection = np.logical_and(true_class, true_label)
        union = np.logical_or(true_class, true_label)
        iou = np.sum(intersection) / np.sum(union)
        accuracy = accuracy_score(true_class, true_label)
        precision = precision_score(true_class, true_label, average='binary')
        recall = recall_score(true_class, true_label, average='binary')
        f1 = f1_score(true_class, true_label, average='binary')
        iou_per_class.append(iou)
        acc_per_class.append(accuracy)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    metrics_dict = {
        'iou':iou_per_class,
        'acc':acc_per_class,
        'precision':precision_per_class,
        'recall':recall_per_class,
        'f1':f1_per_class
    }
    return metrics_dict