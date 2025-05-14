import argparse

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, \
    Normalize, RandomCrop
import torchvision.models as models

from utils.utils import set_all_seed
from utils.utils import target_transforms

from transforms.apply_patch import ApplyPatch

import gzip
import pickle

import os
import cv2
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional
from tqdm import tqdm

from robustbench.utils import load_model


set_all_seed(42)

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())
    

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/imagenet_patch.gz"), 'rb') as f:
    imagenet_patch = pickle.load(f)
patches, targets, info = imagenet_patch

from utils.visualization import show_imagenet_patch
#show_imagenet_patch(patches, targets)

parser = argparse.ArgumentParser(description="Adverserial-Attack")
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--size", type=int, default=50)
parser.add_argument("--model", type=str, default='resnet18')
parser.add_argument("--patch", type=int, default=0)


args = parser.parse_args()
batch_idx = args.batch


## dataset 
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=torchvision.transforms.Compose([]), det=False):
        # with open("ImageNet/val.txt", "r") as f:
        #     self.labels = torch.tensor([int(i.replace("\n", "").split(" ")[1]) for i in f.readlines()])
        self.imgs = glob("ImageNet/ILSVRC2012_img_val/*.JPEG")
        self.bboxes = glob("ImageNet/imagenet_bbox/*")
        self.transforms = transforms
        self.det = det
        
    def __len__(self):
        return 10000
        return len(self.imgs)
    
    def __getitem__(self, idx):
        ## return img, y, bbox
        img = cv2.imread(self.imgs[idx]) / 255
        # get bbox
        with open(self.bboxes[idx]) as f:
            content = f.read()
            xml = ET.fromstring(content)
            bbox = [int(i.text) for i in xml[-1][-1]]  ## xmin, ymin, xmax, ymax
            y = xml[-1][0].text ## name of class
        # y = self.labels[idx]
        
        img = torch.from_numpy(img).permute(2,0,1)
        center = torch.tensor(img.shape[1:]) // 2
        bbox = torch.tensor(bbox).float()
        bbox[::2], bbox[1::2] = bbox[::2] * 256 / img.shape[1], bbox[1::2] * 256 / img.shape[2]
        bbox = torch.tensor([bbox[::2].float().mean(), bbox[1::2].float().mean()]).int()
        tr = (bbox - center) * torch.tensor([256/img.shape[1], 256/img.shape[2]])
        # bbox - torch.cat([center,center])
        return (self.transforms(img), y, bbox) if self.det else (self.transforms(img), y)
        # return torch.from_numpy(img), self.labels[idx], torch.tensor(bbox)
        
        
##================================================================================================================
# Choose an integer in the range 0-9 to select the patch
patch_id = args.patch
patch = patches[patch_id]      # get the chosen patch
target = targets[patch_id]
patch_size = args.size / info['patch_size']

# Instantiate the ApplyPatch module setting the patch and the affine transformation that will be applied
apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                         translation_range=(.0, .0),
                         rotation_range=(-0, 0),
                         scale_range=(patch_size, patch_size)           # scale range wrt image dimensions,
                         )

# For convenience the preprocessing steps are splitted to compute also the clean predictions
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
patch_normalizer = Compose([apply_patch, normalizer])
z = Compose([apply_patch, normalizer])

# Load the data
preprocess = Compose([Resize(256), RandomCrop(224)])    # ensure images are 224x224

dataset = ImageNetDataset(transforms=preprocess, det=True)

data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
x, y, tr = next(iter(data_loader))  # load a mini-batch


x_clean = normalizer(x)
## translations were given to apply_patch with x[i]
x_adv = normalizer(torch.cat([apply_patch(x[i], [0,0]).unsqueeze(0) for i in range(x.shape[0])], dim=0))


## evaluate


## load model
model = None
if args.model == "resnet18":
    model = torchvision.models.resnet18(pretrained=True)
elif args.model == "resnet50":
    model = torchvision.models.resnet50(pretrained=True)
elif args.model == "mobilenetv2":
    model = torchvision.models.mobilenet_v2(pretrained=True)
elif args.model == "mobilenetv3":
    model = torchvision.models.mobilenet_v3(pretrained=True)    

#model = load_model("Wong2020Fast", dataset="imagenet", threat_model='Linf').cuda()
model_large = models.resnet152(pretrained=True)
model.cuda()
model.eval()

seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
seg_model.cuda()
seg_model.eval()


sizes = [2.0, 0.5, 0.3, 0.1] ## we already have 1.0, these will be: 100, 25, 15, 5 respectively
types = ["center", "seg", "fixed", "random"]
#types = ["seg"]
for s in sizes:
  apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                         translation_range=(.0, .0),
                         rotation_range=(-0, 0),
                         scale_range=(s, s)           # scale range wrt image dimensions,
                         )
  for t in types:
      accuracies = [[], []]
      preds = [[], []]
      accuracy_clean = 0.0
      accuracy_adv = 0.0
      for x, y, tr in tqdm(data_loader):
          ## load data
          # x, y = next(iter(data_loader))  # load a mini-batch
          x_clean = normalizer(x).float().cuda()
  
          # sizes.append([(b[2]-b[0])*(b[3]-b[1]) for b in tr])
          
          # here we watn to apply patch with translation as with bbox
          if t == "seg":
            mask = seg_model(x_clean.float())["out"].cpu().clone().detach()
            avg = torch.nn.functional.pad(mask[:,0], (25,25,25,25), value=mask[:,0].max().item()) ## takes background prediction
            avg = torch.nn.functional.avg_pool2d(avg, kernel_size=51, stride=1, padding=0, ceil_mode=True)
            inds = torch.cat([torch.stack(list(torch.where(avg[i] == avg[i].min())), dim=1) for i in range(avg.shape[0])], dim=0)
            tr = inds - torch.tensor([[224, 224]])/2 ## transform from center
            tr = tr.tolist()[::1]
            #tr = torch.zeros_like(tr)
          elif t == "center" or t == "fixed":
            tr = torch.ones_like(tr) * 0 if t == "center" else torch.ones_like(tr) * 80
          elif t == "random":
            tr = torch.randint(-120, 120, tr.shape)
            
          x_adv = normalizer(torch.cat([apply_patch(x[i], tr[i]).unsqueeze(0) for i in range(x.shape[0])], dim=0)) ## center of object
          
          # ind = tr.abs() > 80
          # tr[ind] = 80 * tr[ind]//tr[ind].abs()
          x_adv = normalizer(torch.cat([apply_patch(x[i], tr[i]).unsqueeze(0) for i in range(x.shape[0])], dim=0)).float().cuda() ## center of object
          # x_adv = normalizer(apply_patch(x)).float() 
  
          # Feed the model with the clean images
          if t == "center1":
            output_clean = model_large(x_clean)
          else:
            output_clean = model(x_clean)
          clean_predictions = torch.argmax(output_clean, dim=1).cpu().clone().detach()
  
          # Feed the model with the images corrupted by the patch
          output_adv = model(x_adv)
          adv_predictions = torch.argmax(output_adv, dim=1).cpu().clone().detach()
          
          preds[0].append(clean_predictions)
          preds[1].append(adv_predictions)
  
      #np.save(f"results/preds/{t}_2", preds)
      torch.save(preds, f"results/ablations/{int(s*50)}_{t}_cropped_resnet50.pt")
        
        
        
