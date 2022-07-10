from glob import glob
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
from skimage import transform
import skimage
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
def demosaic_raw(meas):
	tform = transform.SimilarityTransform(rotation=0.00174)
	X = meas.numpy()[0,:,:]
	X = X/np.max(X)
	im1=np.zeros((512,640,4))
	im1[:,:,0]=X[0::2, 0::2]#b
	im1[:,:,1]=X[0::2, 1::2]#gb
	im1[:,:,2]=X[1::2, 0::2]#gr
	im1[:,:,3]=X[1::2, 1::2]#r
	im1=skimage.transform.warp(im1,tform)
	im=im1[6:506,10:630,:]
	#im = im1[128:384,192:448,:]      
	#im = im1[6:131,10:165,:]
	im2=np.zeros((500,620,3))
	im2[:,:,0] = im[:,:,3]
	im2[:,:,1] = 0.5*(im[:,:,2]+im[:,:,1])
	im2[:,:,2] = im[:,:,0]
	im3 = im2[0:500,0:620,:]  	    
	rowMeans = im3.mean(axis=1, keepdims=True)
	colMeans = im3.mean(axis=0, keepdims=True)
	allMean = rowMeans.mean()
	im3 = im3 - rowMeans - colMeans + allMean
	im3 = im3.astype('float32')
	meas = torch.from_numpy(np.array(np.swapaxes(np.swapaxes(im3,0,2),1,2),copy = True)).unsqueeze(0)
	return meas[0,:,:,:]

class Dataset(Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None):
        '''
        :param data_dir: directory where the dataset is kept
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.data_dir = data_dir
        self.wh = 256
        # self.img_list = list()
        # self.msk_list = list()
        # self.input_list = list()
        self.resize = torchvision.transforms.Resize((self.wh,self.wh))    
        self.totensor = torchvision.transforms.ToTensor()
        self.img_list = glob(data_dir+"/*.jpg")
#         with open(osp.join('Lists', dataset + '.txt'), 'r',encoding = 'gb2312') as lines:
# #            lines = lines.decode('utf8')
#             i = 0
#             for line in lines:
#                 i += 1
#                 print('lines',i)
#                 line_arr = line.split()
#                 self.img_list.append(osp.join(self.data_dir, line_arr[0].strip()))
#                 self.msk_list.append(osp.join(self.data_dir, line_arr[1].strip()))
#                 self.input_list.append(osp.join(self.data_dir, line_arr[2].strip()))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        route = os.path.join(self.data_dir, str(idx+4)+".jpg")
        print(route)
        meas = Image.open(route).convert('RGB')        
        # image = Image.open(self.img_list[idx].replace('/home/yxj001/sp0722/Flatnet_quantization-master/DataSets/ICCV-2019-FlatNet-Dataset/','/home/yxj/Research_PHD/')).convert('RGB')
        # label = Image.open(self.msk_list[idx].replace('/home/yxj001/sp0722/Flatnet_quantization-master/DataSets/ICCV-2019-FlatNet-Dataset/','/home/yxj/Research_PHD/Display-Simuated/')).convert('L')
    #    body  = Image.open(self.msk_list[idx].replace('/home/yxj001/sp0722/Flatnet_quantization-master/DataSets/ICCV-2019-FlatNet-Dataset/','/home/yxj/Research_PHD/Display-Simuated/')).convert('L')
     #   detail= Image.open(self.msk_list[idx].replace('/home/yxj001/sp0722/Flatnet_quantization-master/DataSets/ICCV-2019-FlatNet-Dataset/','/home/yxj/Research_PHD/Display-Simuated/')).convert('L')
 #       name  = self.msk_list[idx].replace('/home/yxj001/sp0722/Flatnet_quantization-master/DataSets/ICCV-2019-FlatNet-Dataset/Display-Captures/mask/','')
 #       print(torch.tensor(label).shape)
        meas = self.totensor(meas)
       # if self.transform is not None:
       # [image, label] = self.transform(image, label)
        # image = self.resize(image)
        # label = self.resize(label)            		
      #  body= self.resize(body)
      #  detail= self.resize(detail)
        
        meas = demosaic_raw(meas)
        # image = self.totensor(image)
        # label = self.totensor(label)
       # body= self.totensor(body)
       # detail= self.totensor(detail)
                        
#        [label,label] = self.transform(label,label)            
#        return image, label, meas, body, detail
        return meas