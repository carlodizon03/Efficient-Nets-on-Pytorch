from tensorboardX import SummaryWriter
from datetime import datetime 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
class Logger(object):

    def __init__(self, path):
        self.timestamp = datetime.now()
        self.str_timestamp = str("{0}-{1}-{2}--{3}-{4}-{5}/".format(self.timestamp.month, self.timestamp.day,self.timestamp.year,self.timestamp.hour,self.timestamp.minute,self.timestamp.second))
        self.log_dir =  os.path.join(path, self.str_timestamp)
        self.writer = SummaryWriter(self.log_dir)
    
    def train_loss_per_epoch(self, value, step):
        self.writer.add_scalar('Loss/training_per_epoch', value, step)
        self.writer.flush()

    def train_loss(self, value, step):
        self.writer.add_scalar('Loss/training', value, step)
        self.writer.flush()

    def train_mloss(self, value, step):
        self.writer.add_scalar('mLoss/training', value, step)
        self.writer.flush()
    
    def train_acc(self, value, step):
        self.writer.add_scalar('Accuracy/training', value, step)
        self.writer.flush()
    
    def train_macc(self, value, step):
        self.writer.add_scalar('mAccuracy/training', value, step)
        self.writer.flush()
    

    def train_miou(self,miou,step):
        self.writer.add_scalar('mIoU/training', miou, step)
        self.writer.flush()

    def train_iou_per_class(self,iou,step):
        print(iou)
        cloud, shadow, _ = iou
        self.writer.add_scalars('IoU/training', {'cloud':cloud,'shadow':shadow}, step)
        self.writer.flush()

    def val_loss(self,value,step):
        self.writer.add_scalar('Loss/validation', value, step)
        self.writer.flush()
    
    def val_mloss(self,value,step):
        self.writer.add_scalar('mLoss/validation', value, step)
        self.writer.flush()
    
    def val_acc(self, value, step):
        self.writer.add_scalar('Accuracy/validation', value, step)
        self.writer.flush()
    
    def val_macc(self, value, step):
        self.writer.add_scalar('mAccuracy/validation', value, step)
    
        self.writer.flush()
    def val_loss_per_epoch(self, value, step):
        self.writer.add_scalar('Loss/validation_per_epoch', value, step)
        self.writer.flush()
        
    def val_dsc(self,value,step):
        self.writer.add_scalar('Loss/DSC_per_Volume',value,step)
        self.writer.flush()

    def val_miou(self,miou,step):
        self.writer.add_scalar('mIoU/validation',miou,step)
        self.writer.flush()

    def val_iou_per_class(self,iou,step):
        cloud, shadow, _ = iou
        
        self.writer.add_scalars('IoU/validation', {'cloud':cloud,'shadow':shadow}, step)
        self.writer.flush()

    def val_acc(self,value,step):
        self.writer.add_scalar('Accuracy/validation', value, step)
        self.writer.flush()

    def display_train_batch(self,images,masks,preds,step):
        batch_shape = images.shape
        img_batch   = np.zeros(batch_shape)
        for idx,_ in enumerate(images):
            img_batch[idx, 0]   = images[idx]
            img_batch[idx, 1]   = masks[idx]
            img_batch[idx, 2]   = preds[idx]
        self.writer.add_images = ('Training_Prediction', img_batch, step)
        self.writer.flush()

    def display_val_batch(self,images,masks,preds,epoch):

        images = images.to('cpu')
        masks = masks.to('cpu')
        preds = preds.to('cpu')
        image_grid  = torchvision.utils.make_grid(images)
        mask_grid   = torchvision.utils.make_grid(masks)
        pred_grid   = torchvision.utils.make_grid(preds)
        self.writer.add_image('Validation/image',image_grid,epoch)
        self.writer.add_image('Validation/mask',mask_grid,epoch)
        self.writer.add_image('Validation/prediction',pred_grid,epoch)
        self.writer.flush()
    
    def model_graph(self,model,input_image):
        self.writer.add_graph(model,input_image,verbose=True)
        self.writer.flush()
        self.writer.close()

    def close(self):
        self.writer.close()