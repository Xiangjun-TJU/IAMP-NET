#trai_plus
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from time import time
import math
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from torch_vgg import Vgg16
#from dataloader_plus import*
import matplotlib.pyplot as mp
import datasets
import cv2
#from skimage.measure import compare_ssim as ssim
from torch.autograd import Variable
#from dual_path_unet import dual_path_unet as DUnet
#from numpy.fft import fft, ifft
#from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
parser = ArgumentParser(description='IAMPU-Net')

parser.add_argument('--start_epoch', type=int, default=200, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=7, help='phase number of ISTA-Net')  #9#
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='new_model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--init_matrix_type',type=str, default='Random Toeplitz', help='initized matrix type')

parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default=" ", type=str, help="path to pretrained model (default: none)")
#validation ground truth filenames

args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Training_data_Name = 'Training_Data_Img91.mat'
#Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
#Training_labels = Training_data['labels']
def psnr(img1, img2):
	img1.astype(np.float32)
	img2.astype(np.float32)
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return 100
	PIXEL_MAX = 1.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ssim(img1, img2):
  C1 = (0.01)**2
  C2 = (0.03)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean() 
def load_meas_matrix():
    WL = np.zeros((500,256,1))
    WR = np.zeros((620,256,1))
    d = sio.loadmat('flatcam_prototype2_calibdata.mat') ##Initialize the weight matrices with transpose
    phil = np.zeros((500,256,1))
    phir = np.zeros((620,256,1))

    pl = sio.loadmat('phil_toep_slope22.mat')
    pr = sio.loadmat('phir_toep_slope22.mat')
    WL[:,:,0] = pl['phil'][:,:,0]
    WR[:,:,0] = pr['phir'][:,:,0] 
    #if args.init_matrix_type=='':

    WL = WL.astype('float32')   #  Pseudo inverse   WL
    WR = WR.astype('float32')   #  Pseudo inverse   WR  

    phil[:,:,0] = d['P1b']
    phir[:,:,0] = d['Q1b']
    phil = phil.astype('float32')   #  phiL
    phir = phir.astype('float32')   #  phiR

    return phil,phir,WL,WR

class initial_inversion2(nn.Module):
    def __init__(self):
        super(initial_inversion2, self).__init__()
    def forward(self,meas,WL,WR):
        x0=F.leaky_relu(torch.matmul(torch.matmul(meas[:,0,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x1=F.leaky_relu(torch.matmul(torch.matmul(meas[:,1,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x2=F.leaky_relu(torch.matmul(torch.matmul(meas[:,2,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X_init=torch.cat((x0,x1,x2),3)
        X_init = X_init.permute(0,3,1,2)
        return X_init
# the gradient iteration block
class initial_inversion(nn.Module):
	def __init__(self):
		super(initial_inversion,self).__init__()
	#	self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

	def forward(self,Xinp,Z,phil,phir):
		y0=F.leaky_relu(torch.matmul(torch.matmul(Z[:,0,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
		y10=F.leaky_relu(torch.matmul(torch.matmul(Z[:,1,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
		y11=F.leaky_relu(torch.matmul(torch.matmul(Z[:,2,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
	#	y2=F.leaky_relu(torch.matmul(torch.matmul(Z[:,3,:,:],phir[:,:,3]).permute(0,2,1),phil[:,:,3]).permute(0,2,1).unsqueeze(3))
		Y_init=torch.cat((y0,y10,y11),3)
		y_init = Y_init.permute(0,3,1,2)    
		R = y_init + Xinp  
		return R
      
 #Attention    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes , 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
      
                  
# Define IAMP-Net Block
class BasicBlock(torch.nn.Module):
	def __init__(self,phil,phir):
		super(BasicBlock, self).__init__()
		self.soft_thr = nn.Parameter(torch.Tensor([0.0005]))
		self.tau = nn.Parameter(torch.Tensor([0.25]))
		self.PhiL = nn.Parameter(torch.tensor(phil),requires_grad=False)
		self.PhiR = nn.Parameter(torch.tensor(phir),requires_grad=False)
		self.eta = nn.Parameter(torch.Tensor([2]))
		self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 3, 3, 3)))
		self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))		
		self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))		
		self.conv4_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(3, 32, 3, 3)))
		self.gradLayer = initial_inversion()
		self.ini_inversion = initial_inversion2()
	def forward(self, meas, x, Z):
		R = self.gradLayer(x,Z,self.PhiL,self.PhiR)
		x = F.conv2d(R, self.conv1_forward, padding=1)
		R1 = F.relu(x)
		x = F.conv2d(R1, self.conv2_forward, padding=1)
		x = x+R1
		R2 = F.relu(x)
		x = F.conv2d(R2, self.conv3_forward, padding=1)
		x = x+R2
		x = F.relu(x)
		x_forward = F.conv2d(x, self.conv4_forward, padding=1)
		#x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
		mu1 = torch.sign(F.relu(torch.mul(torch.abs(x_forward)-self.soft_thr,self.soft_thr*self.eta-torch.abs(x_forward))))
		mu2 = torch.sign(F.relu(torch.abs(x_forward)-self.soft_thr*self.eta))

		L = torch.mul(mu1,torch.mul(self.eta/(self.eta-1),torch.mul(torch.abs(x_forward)-self.soft_thr,torch.sign(x_forward))))+torch.mul(mu2,x_forward)
		x = F.conv2d(L, self.conv1_backward, padding=1)
		L1 = F.relu(x)
		x = F.conv2d(L1, self.conv2_backward, padding=1)
		x = x+L1
		L2 = F.relu(x)
		x = F.conv2d(L2, self.conv3_backward, padding=1)
		x = x+L2
		x = F.relu(x)
		x_backward = F.conv2d(x, self.conv4_backward, padding=1)

		x = F.conv2d(x_forward, self.conv1_backward, padding=1)
		x = x+L
		L1 = F.relu(x)
		x = F.conv2d(L1, self.conv2_backward, padding=1)
		x = x+L1
		L2 = F.relu(x)
		x = F.conv2d(L2, self.conv3_backward, padding=1)
		x = x+L2
		x = F.relu(x)
		x_est = F.conv2d(x, self.conv4_backward, padding=1)

		Z = meas- self.ini_inversion(x_backward,self.PhiL.permute(1,0,2),self.PhiR.permute(1,0,2))+torch.mul(self.tau,Z)

		symloss = x_est - R

		return [x_backward, Z, symloss]

# Define IAMP-Net
class IAMPNet(torch.nn.Module):
	def __init__(self, phil, phir, WL, WR, LayerNo):
		super(IAMPNet, self).__init__()
		onelayer = []
		self.LayerNo = LayerNo
		self.PhiL = nn.Parameter(torch.tensor(phil))
		self.PhiR = nn.Parameter(torch.tensor(phir))
		self.WL = nn.Parameter(torch.tensor(WL))
		self.WR = nn.Parameter(torch.tensor(WR))
		for i in range(LayerNo):
			onelayer.append(BasicBlock(phil,phir))

		self.fcs = nn.ModuleList(onelayer)
		self.ini_inversion = initial_inversion2()
#		self.enhancement = DUnet(4,3,32) 
	def forward(self, meas):
		x = self.ini_inversion(meas,self.WL,self.WR)


		Y = self.ini_inversion(x,self.PhiL.permute(1,0,2),self.PhiR.permute(1,0,2))

		Z = meas - Y		
		x_init = x
		layers_sym = []   # for computing symmetric loss
		for i in range(self.LayerNo):
			[x, Z, layer_sym] = self.fcs[i](meas, x, Z)
			layers_sym.append(layer_sym)			
#		x_final = self.enhancement(x_init,x)
		x_final = x
		return [x_final, layers_sym]
   

phil,phir,WL,WR = load_meas_matrix()
model = IAMPNet(phil, phir, WL, WR,layer_num)
vgg = Vgg16(requires_grad=False)
criterion = nn.MSELoss()
#criterion1 = MS_SSIM_L1_LOSS()
#model = nn.DataParallel(model)
model = model.to(device)
Vgg = vgg.to(device)
criterion = criterion.to(device)




print_flag = 0   # print parameter number

if print_flag:
	num_count = 0
	for para in model.parameters():
		num_count += 1
		print('Layer %d' % num_count)
		print(para.size())



class RandomDataset(Dataset):
	def __init__(self, data, length):
		self.data = data
		self.len = length

	def __getitem__(self, index):
		return torch.Tensor(self.data[index, :]).float()

	def __len__(self):
		return self.len
########################################################

val_set = datasets.Dataset('./flat', 'flat', transform=None)

val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,num_workers=4, pin_memory=True)

def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
	cycle = np.floor(1 + iteration/(2  * stepsize))
	x = np.abs(iteration/stepsize - 2 * cycle + 1)
	lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
	return lr

def adjust_learning_rate(optimizer, epoch):
	if epoch in range(50):
		lr = args.learning_rate
	elif epoch in range(50,100):
		lr = 0.5*args.learning_rate/(epoch - 49)
	elif epoch in range(100,150):
		lr = 0.25*args.learning_rate/(epoch - 99)
	else:
		lr = 0.125*args.learning_rate/(epoch - 149)
	return lr
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

model_dir = "./%s/ResCS_IAMP_Net__Dunet_%d_group_%d_ratio_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, cs_ratio, learning_rate)
log_file_name = "./%s/ResLog_CS_IAMP_Dunet_%d_group_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
	os.makedirs(model_dir)
if start_epoch > 1:
	pre_model_dir = model_dir
	model.load_state_dict(torch.load("net_params_100.pth", map_location=device))
#iTlos = np.zeros((1,5000)) 
#fTlos = np.zeros((1,5000))

# Training loo
with torch.no_grad():
	PSNR_SUM = 0
	SSIM_SUM = 0
	total_time = 0
	for iteration1, batch_data in enumerate(val_data_loader, 1):
		input = batch_data
		meas = input.cuda()
		
		meas = torch.autograd.Variable(meas)
		
		[x_final, loss_layers_sym] = model(meas)


		Prediction_value = x_final.cpu().data.numpy()
		




		X_rec=np.swapaxes(np.swapaxes(Prediction_value[0,:,:,:],0,2),0,1)
		X_rec=255*(X_rec-np.min(X_rec))/(np.max(X_rec)-np.min(X_rec))
		Xrec = X_rec.astype(np.uint8)

	
		# X=(X-np.min(X))/(np.max(X)-np.min(X))


		
		#Meas=(Meas-np.min(Meas))/(np.max(Meas)-np.min(Meas))


#		rec_PSNR = psnr(X_rec.astype(np.float64), X.astype(np.float64))

#			if iteration < 102:
#					for id in range(0,3):
#							rec_SSIM += ssim(X_rec[:,:,id].astype(np.float64), X[:,:,id].astype(np.float64), data_range=1)
#		PSNR_SUM += rec_PSNR
#					SSIM_SUM += rec_SSIM 
		cv2.imwrite(os.path.join("./rec", str(iteration1)+".jpg"),X_rec[:,:,::-1])
		# mp.imsave('./Test_IAMP/gt/'+str(iteration1)+'_gt.png',X)
		# mp.imsave('./Test_IAMP/meas/'+str(iteration1)+'_meas.png',X)      
		output_data = "PSNR: %.4f" % (PSNR_SUM/iteration1)
	#print(output_data)
#sio.savemat('result.mat', mdict={'init_loss': iTlos, 'final_loss': fTlos})




