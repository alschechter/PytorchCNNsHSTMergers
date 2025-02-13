import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE, Isomap
import glob
from astropy.visualization import simple_norm
#import os
#from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchsummary import summary


# Load the saved model state
checkpoint = torch.load('ResNet18CNN.pth', map_location=torch.device('cpu'))
model = models.resnet18(pretrained=True)
#Set up the model for inference
model.fc = nn.Linear(512, 1)  # Ensure the output layer is the same

# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cpu')  # Ensure the model is on the correct device (GPU/CPU)
model.eval()  # Switch to evaluation mode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from BinaryMergerDataset import BinaryMergerDataset, get_transforms
path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
BATCH_SIZE = 64

test_mergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = True, transform = get_transforms(aug=False), codetest=False)
test_nonmergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = False, transform = get_transforms(aug=False), codetest=False)

test_dataset_full = torch.utils.data.ConcatDataset([test_mergers_dataset_orig, test_nonmergers_dataset_orig])
test_dataloader = DataLoader(test_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE)#num workers used to be 4

all_labels = []
all_preds = []
all_names = []

with torch.no_grad():  # No need to track gradients during inference
    for images, labels, names in tqdm(test_dataloader):
        #print(type(names))
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        # Forward pass
        outputs = model(images)
        # print(type(outputs))
        # print(outputs.shape)
        probabilities = torch.sigmoid(outputs)
        pred = (probabilities > 0.5).float()  # Convert to binary (0 or 1)
        pred = pred.to(device=device, dtype=torch.float32)
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_names.extend(names)
        
# 4. Compute accuracy or other evaluation metrics (e.g., confusion matrix)
# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.squeeze(np.array(all_preds))
all_names = np.array(all_names)



# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_names = np.array(all_names)

#class_for_gradcam = 'nonmerger'
class_for_gradcam = 'merger'

if class_for_gradcam == 'merger':
    targetclass = 0
else:
    targetclass = 1
    
print('targetclass', targetclass)
#### GradCAM with help from github and chat GPT
# We have to specify the target we want to generate the CAM for.
# targets = [ClassifierOutputTarget(0)] #target class is mergers
# print(targets)
targets = [ClassifierOutputTarget(targetclass)] #target class is mergers
# print(targets)
print('ClassifierOutputTarget', ClassifierOutputTarget(targetclass), type(ClassifierOutputTarget(targetclass)))

targets_GradCAM = [model.layer4[-1]]
layer_names_GradCAM = ['layer4_n1']
# Select the first validation image to visualize Grad-CAM
sample_index_choices = np.arange(0, 2)#len(test_dataloader.dataset))  # Change this index to visualize different images
for sample_index in sample_index_choices:
    sample_image = test_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
    sample_label = test_dataloader.dataset[sample_index][1]
    sample_name = test_dataloader.dataset[sample_index][2]
    #print(str(sample_name))
    # Move to device
    sample_image = sample_image.to(torch.float32).to(device)

    
    # Get the original image for visualization
    original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
    predicted_class = (torch.sigmoid(model(sample_image)) > 0.5).int().item()
    predicted_label = f"Predicted: {predicted_class}, True: {sample_label}"


    # Generate the Grad-CAM visualization
    #targets = [ClassifierOutputTarget(int(sample_label))]  # Use the true label as target
    counter = 0
    for target_layer_GradCAM in targets_GradCAM:    
        layername = layer_names_GradCAM[counter]
        counter += 1
        #print('target', target_layer_GradCAM)
        with GradCAM(model=model, target_layers=[target_layer_GradCAM]) as cam:
            print('hello??')
            #print('shapes', np.shape(sample_image), np.shape(targets))
            grayscale_cam = cam(input_tensor=sample_image, targets=targets)
            print('hi?')
            print(np.shape(grayscale_cam)) 
            grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
            print('greyscale2', grayscale_cam)
        # Convert the input image to a format suitable for visualization
        rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
        #grayscale_cam = grayscale_cam#.numpy()

        # Visualize the Grad-CAM output
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        #make labels words
        if sample_label == 1.:
            sample_label_word = 'nonmerger'
        else:
            sample_label_word = 'merger'
            
        if predicted_class == 1.:
            predicted_class_word = 'nonmerger'
        else:
            predicted_class_word = 'merger'
        # Plot the Grad-CAM
        #print(np.min(original_img), np.max(original_img)) #0s and 1s cause it's normalized 
        #print(np.shape(np.sum(original_img, axis = 2)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
        # Original image subplot
        im1 = ax1.imshow(np.sum(original_img, axis = 2), cmap = 'magma', norm = 'log') #assign image for colorbar
        ax1.set_title(f"Original Image - True: {sample_label_word}")
        ax1.axis('off')  # Turn off axis
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', norm=LogNorm(vmin = 0, vmax = 1)) #vmin=np.min(original_img), vmax=np.max(original_img)
        cbar1.set_label('Intensity (log scale)')
        #GradCAM
        im2 = ax2.imshow(visualization)
        #ax2.title(f'Grad-CAM for Validation Image {sample_index} Layer {layername} (True Label: {sample_label})')
        ax2.set_title(f"Predicted: {predicted_class_word}")
        ax2.axis('off')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
        cbar2.set_label('Grad-CAM Intensity')
        
        plt.tight_layout()
        plt.savefig(f"/gradcam/test/predicted_label_' + class_for_gradcam + '/gradcam_test_image_{sample_name}.png", dpi=300)
        #plt.show()
        plt.clf()
        plt.close()
            
