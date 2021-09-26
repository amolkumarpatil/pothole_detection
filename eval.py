# pytorch
import torch
import torchvision
from torch.utils.data import DataLoader

# general
import os
from tqdm import tqdm
import my_custom_transforms as mtr
from dataloader_rgbdsod import RgbdSodDataset
from PIL import Image
from nn.RgbdNet import MyNet as RgbdNet
import cv2
import numpy as np
from test_classify import Classify

size = (224, 224)
datasets_path = './dataset/pothole600/'
test_datasets = ['testing']
pretrained_models = {'RgbdNet': './weights/[2021-]_[RgbdNet]/rgbd.pth'}
result_path = './output/'
os.makedirs(result_path, exist_ok=True)
for tmp in ['results']:
    os.makedirs(os.path.join(result_path, tmp), exist_ok=True)
    for test_dataset in test_datasets:
        os.makedirs(os.path.join(result_path, tmp, test_dataset), exist_ok=True)

model_rgbd = RgbdNet()

model_rgbd.load_state_dict(torch.load(pretrained_models['RgbdNet'], map_location=torch.device('cpu'))['model'],
                           strict=False)

model_rgbd.eval()

transform_test = torchvision.transforms.Compose([mtr.Resize(size), mtr.ToTensor(),
                                                 mtr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                               elems_do=['img'])])

test_loaders = []
for test_dataset in test_datasets:
    val_set = RgbdSodDataset(datasets_path + test_dataset, transform=transform_test)
    test_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True))

for index, test_loader in enumerate(test_loaders):
    dataset = test_datasets[index]
    print('Test [{}]'.format(dataset))

    for i, sample_batched in enumerate(tqdm(test_loader)):
        input, gt = model_rgbd.get_input(sample_batched), model_rgbd.get_gt(sample_batched)

        with torch.no_grad():
            output_rgbd = model_rgbd(input)

        result_rgbd = model_rgbd.get_result(output_rgbd)

        id = sample_batched['meta']['id'][0]
        gt_src = np.array(Image.open(sample_batched['meta']['gt_path'][0]).convert('L'))

        result_rgbd = (cv2.resize(result_rgbd, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) * 255).astype(
            np.uint8)

        Image.fromarray(result_rgbd).save(os.path.join(result_path,'results',dataset,id+'.png'))

        # classify image
        opencv_image = result_rgbd.reshape((result_rgbd.shape[0], result_rgbd.shape[1], 1))
        classify = Classify(opencv_image)
        out_img = classify.detect_n_draw_bb(opencv_image, sample_batched['meta']['img_path'][0])
        out_path = os.path.join(result_path,id+'.png')
        cv2.imwrite(out_path, out_img)
print("Processing completed......")
print("Processing completed......")
