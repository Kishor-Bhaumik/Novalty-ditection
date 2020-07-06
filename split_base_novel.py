import time
start= time.time()

import torch
from torchvision import datasets
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from torch.backends import cudnn
from multiprocessing import Pool
import cv2 
from torchvision import transforms, utils
from torch.utils import data
from sklearn.model_selection import train_test_split
import random 
  

import os, glob
import csv

SEED=42
cudnn.benchmark = True
trainL=[0,1,2,3,4,5,6,7]
testL= [8,9]
use_cuda = torch.cuda.is_available()
device_id=0,5,6,7
bs=16*len(list(device_id))
dev="cuda:"+ str(list(device_id)).strip('[]')
device = torch.device(dev if use_cuda else "cpu")
#device = torch.device("cuda:0" if use_cuda else "cpu")
class_total=len(trainL)

random.seed(SEED)
torch.cuda.manual_seed(SEED)

with open('data/all_cifar.npy', 'rb') as f:
    q= np.load(f)

def getData(Q):
    
    for i,val in enumerate(Q):

        if i is 0: loader= q[val, :, :]
        else :
            add=q[val, :, :]
            loader=np.append(loader,add,axis=0)
    
    return loader
 
train=getData(trainL)


x_train, x_test, y_train, y_test = train_test_split(train[:,1:], train[:, 0], test_size=0.2, random_state=42)

np.save("data/novel_test.npy",x_test)
np.save("data/novel_label.npy", y_test)

transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class CIFAR(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i, :]
        data = data.astype(np.uint8).reshape(32,32,3)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
 

train_data = CIFAR(x_train, y_train, transform)
trainloader = DataLoader(train_data, batch_size=bs, shuffle=True ,pin_memory=True)


PATH='/home/akmmrahman/Drakmmrahman/kishore/cam/load/vgg16_imagenet.pth.tar'

model_vgg = torch.load(PATH)
class custom_vgg (nn.Module):
    def __init__(self):
        super(custom_vgg,self).__init__()
        self.features = model_vgg.features
        self.features2= model_vgg.features[:15]
        #self.features_vae = model.features
        #self.bn2 =nn.BatchNorm2d(512)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=(7,7))
        self.drop = nn.Dropout(0.7)
        self.fc = nn.Linear(512,class_total)
#         self.soft = nn.Softmax(dim = 1)
        
    def forward(self,x):
        c1 = self.features(x)
        Cnew=self.features2(x)
        c3 = self.global_avg_pool(c1)
        c4 = c3.view(-1,512)
        c5 = self.drop(c4)
        out = self.fc(c5)
#         out = self.soft(c6)
        return F.relu(Cnew), out 

class VAE(nn.Module):
    def __init__(self,inp,hidden,latent_variable_size):
        super(VAE, self).__init__()

        self.INP=inp
        self.hidden = hidden
        self.latent_variable_size = latent_variable_size
        self.bn = nn.BatchNorm1d(16384)
        self.fc1 = nn.Linear(inp, hidden)
        self.fc21 = nn.Linear(hidden, latent_variable_size)
        self.fc22 = nn.Linear(hidden, latent_variable_size)
        self.fc3 = nn.Linear(latent_variable_size, hidden)
        self.fc4 = nn.Linear(hidden, inp)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m)
            except:
                normal_init(block)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        v=torch.sigmoid(self.fc4(h3))
        return v.reshape(v.size(0),256,8,8)

    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        x=self.bn(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    
def loss_function(recon_x, x, mu, logvar):
    #BCE =F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE=F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #kld= crit(recon_x, x)
    
    return BCE + KLD ,BCE,KLD



model =custom_vgg()
model = nn.DataParallel(model, device_ids=list(device_id))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
vae_loss=[]
epoch_loss=[]
#model_vae = VAE(inp=512*7*7,hidden=5000,latent_variable_size=1000).to(device)
#model_vae = VAE(512*7*7,5000,100)
model_vae = VAE(256*8*8,5000,100)
#model_vae.apply(init_weights)
model_vae = nn.DataParallel(model_vae, device_ids=list(device_id))
model_vae.to(device)

optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-3)
#optimizer_vae = optim.SGD(model_vae.parameters(), lr=1e-3)
max_epochs = 20
model.train()
model_vae.train()

train_loss_min = np.Inf 
for epoch in range(max_epochs):
    train_loss = 0.0
    cn = 0
    epoch_losssss = 0
    acc = 0
    vae_train_loss=0
    bnce=0
    kldl=0
    cl=0
    recl=0
    # Training
    print("Epoch :", epoch+1)

    #itr = (training_set.__len__()) // batch_size
    #print(type(training_generator))
    #print(itr,len(training_generator.dataset))
    
    if __name__ == "__main__":
        for local_batch, local_labels in trainloader:
            #print(local_labels[:1,:])
            cn = cn+1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device,non_blocking=True), local_labels.to(device,non_blocking=True)
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            optimizer.zero_grad()
            optimizer_vae.zero_grad()
         
            # forward + backward + optimize
            c1,outputs = model(local_batch)
            
            c1=c1.detach().to(device,non_blocking=True)
 
            recon_batch, mu, logvar = model_vae(c1)
            recon_loss,bce,kld= loss_function(recon_batch,c1 , mu, logvar) #.item()
            #print(VAE_LOSS.item())
            cl_loss=criterion(outputs, local_labels)
            loss_total = cl_loss+recon_loss
            loss,loss_vae= loss_total,loss_total
            
            #vae_loss.append((epoch+1,cn,VAE_LOSS.item(),cl_loss.item(),bce.item(),kld.item()))
            loss.backward(retain_graph=True)
            loss_vae.backward()

            vae_train_loss+=loss_vae.item()*local_batch.size(0)
            kldl+=kld.item()*local_batch.size(0)
            bnce+=bce.item()*local_batch.size(0)
            cl+=cl_loss.item()*local_batch.size(0)
            recl+=recon_loss.item()*local_batch.size(0)

            optimizer.step()
            optimizer_vae.step()
    if vae_train_loss <= train_loss_min:
        
        for filename in glob.glob("model_*"):
            os.remove(filename)
        torch.save(model.state_dict(), 'model_vgg_best_epoch'+str(epoch+1)+'.pt')
        torch.save(model_vae.state_dict(), 'model_vae_best_epoch'+str(epoch+1)+'.pt')
        train_loss_min = vae_train_loss

    L=len(trainloader.dataset)
    avg_loss_recl=recl/ L
    avg_loss_cl= cl/L
    avg_loss_kld= kldl/L
    avg_loss_bce=bnce/L
    #print(" vae loss =", avg_loss)
    epoch_loss.append((epoch+1,avg_loss_recl,avg_loss_cl,avg_loss_bce,
                       avg_loss_kld))
    #print(" vae loss =", vae_train_loss/ 5)
    #break


with open('cifar_vae_loss_final.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(["Epoch","classifaction_loss","reconstruction_loss","mse_loss","KLD_loss"])
    for epoch,vloss,clloss,mse,kld in epoch_loss:
        employee_writer.writerow([str(epoch),str(clloss),str(vloss),str(mse),str(kld)])
        
qw=time.time()-start
print(qw/3600)