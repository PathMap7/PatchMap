import argparse
import torch
import gzip
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
import json

set_all_seed(42)

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())
    

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/imagenet_patch.gz"), 'rb') as f:
    imagenet_patch = pickle.load(f)
patches, targets, info = imagenet_patch

from utils.visualization import show_imagenet_patch


parser = argparse.ArgumentParser(description="Adverserial-Attack")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=1)
parser.add_argument("--size", type=int, default=50)
parser.add_argument("--model", type=str, default='resnet18')
parser.add_argument("--patch", type=int, default=0)


args = parser.parse_args()
print(f"arguments:{args.__dict__}")


## dataset 
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=torchvision.transforms.Compose([]), det=False):
        # with open("ImageNet/val.txt", "r") as f:
        #     self.labels = torch.tensor([int(i.replace("\n", "").split(" ")[1]) for i in f.readlines()])
        self.imgs = glob("ImageNet/ILSVRC2012_img_val/*.JPEG")
        self.bboxes = glob("ImageNet/imagenet_bbox/*")
        self.transforms = transforms
        self.det = det
        self.inds = torch.load("assets/imagenet_inds.pth")
        
    def __len__(self):
        return len(self.inds)
        return len(self.imgs)
    
    def __getitem__(self, idx):
        idx = self.inds[idx]
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
        
        
def load_data(all_values):
    cls = list(all_values[0].items())[0][1]
    n = all_values[1]['(0, 0)'].shape[-1]
    scale = 2
    size = 224
    arr = torch.zeros(size//scale,size//scale, n)
    arr_soft = torch.zeros(size//scale,size//scale, n)
    for k in (all_values[1].keys()):
        inds = torch.tensor(json.loads(k.replace("(","[").replace(")","]")))
        inds = (112 + inds) // scale ## for 112X112 values
        arr[tuple(inds)] = all_values[1][k]
        arr_soft[tuple(inds)] = all_values[2][k]
    arr_soft, arr = arr_soft.permute(2,0,1).clone().detach(), arr.permute(2,0,1).clone().detach()
    return arr, arr_soft

class ImageLocationDataset(torch.utils.data.Dataset):
    def __init__(self, data=[]):
        data = [load_data(i) for i in (data)]
        self.arr, self.arr_soft = [torch.cat([i[j] for i in data], dim=0) for j in range(2)]

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        return self.arr_soft[idx], self.arr[idx]
        
        
        
##================================================================================================================
# Choose an integer in the range 0-9 to select the patch
patch_id = args.patch
patch = patches[patch_id]      # get the chosen patch
target = targets[patch_id]
patch_size = args.size / info['patch_size']

batch_size = 10

# Instantiate the ApplyPatch module setting the patch and the affine transformation that will be applied
apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                         translation_range=(.0, .0),
                         rotation_range=(-0, 0),
                         scale_range=(patch_size, patch_size)           # scale range wrt image dimensions,
                         )
print("patch size:" + str(patch_size))

# For convenience the preprocessing steps are splitted to compute also the clean predictions
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
patch_normalizer = Compose([apply_patch, normalizer])
z = Compose([apply_patch, normalizer])

# Load the data
preprocess = Compose([Resize(256), RandomCrop(224)])    # ensure images are 224x224

dataset = ImageNetDataset(transforms=preprocess, det=True)
print(f"{len(dataset)} examples")

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

## translations were given to apply_patch with x[i]


## evaluate

model = models.resnet50(pretrained=True)
model.cuda()
model.eval()

seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
seg_model.cuda()
seg_model.eval()


# sizes = []
types = ["center", "seg", "fixed", "random"]
x, y, tr = next(iter(data_loader))
#x, y, tr = next(iter(data_loader))

#types = ["seg"]

accuracies = [[], []]
preds = [{},{}, {}, {}]
preds_total = []
accuracy_clean = 0.0
accuracy_adv = 0.0
it = iter(data_loader)
gpu = 0
start = args.start // batch_size
end = args.end // batch_size


'''
y_confidance = []
y_pred = []
for x, y, tr in tqdm(data_loader):
  x = x.float().cuda()
  pred = model(x).cpu()
  pred = torch.nn.functional.softmax(pred, dim=1)
  predicted = pred.argmax(dim=1)
  pred = pred[:,pred.argmax(dim=1)]
  pred = torch.tensor([pred[i,i] for i in range(10)])
  y_confidance.append(pred.cpu().clone().detach())
  y_pred.append(predicted.cpu().clone().detach())

#torch.save(torch.cat(y_confidance, dim=0), "dataset/baseline_confidance.pt")
torch.save(torch.cat(y_pred, dim=0), "dataset/baseline_pred.pt")
exit(0)
'''

#for _ in range(start):
#  x, y, tr = next(it)
#bar = tqdm(range(start, end))
os.makedirs("temp_dataset/effect_stats")
bar = tqdm(data_loader)
for b_idx in bar:
  x, y, tr = next(it)
  preds = [{},{}, {}, {}]
  for i in range(-112,112,2):
      for j in range(-112,112,2):
        ## load data
        # x, y = next(iter(data_loader))  # load a mini-batch
        x_clean = normalizer(x).float().cuda()
  
        x_adv = normalizer(torch.cat([apply_patch(x[ind], (j, i)).unsqueeze(0) for ind in range(x.shape[0])], dim=0)).float().cuda() ## center of object
        # x_adv = normalizer(apply_patch(x)).float() 
    
        # Feed the model with the clean images
        
        output_clean = model(x_clean)
        clean_predictions = torch.argmax(output_clean, dim=1).cpu().clone().detach()
    
        # Feed the model with the images corrupted by the patch
        output_adv = model(x_adv)
        adv_predictions = torch.argmax(output_adv, dim=1).cpu().clone().detach()
        adv_scores = torch.softmax(output_adv, dim=1).cpu().clone().detach()[:,clean_predictions].diag()
        #adv_predictions = torch.argmax(output_adv, dim=1).cpu().clone().detach()
        
        preds[0][f"{i,j}"] = clean_predictions
        preds[1][f"{i,j}"] = adv_predictions
        preds[2][f"{i,j}"] = adv_scores ## scores for best class
        #preds[3]['x'] = x_clean.cpu()
        bar.set_description(f"({i,j})/(112,112)")
        
  preds_total.append(preds)
  torch.save(preds_total, f"temp_dataset/effect_stats/effect_predictions_{patch_id}_{args.size}.pt")
  
directory_name = f"dataset/effect/{patch_id}/{args.size}/"
os.makedirs(directory_name)
ilds = ImageLocationDataset(preds_total)
print("saving dataset")
for i in tqdm(range(len(ilds))):
    img_name = ds.imgs[i].split("_")[-1].split(".")[0]
    z = ilds[i]
    z = torch.cat([z[1], z[0]], dim=0).numpy()
    np.save(f"{directory_name}/{img_name}.npy", z)



      
      
      
