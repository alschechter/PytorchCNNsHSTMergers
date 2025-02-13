
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
#from sklearn.manifold import TSNE, Isomap
import glob
from astropy.visualization import simple_norm
#import os
#from torchvision.io import read_image
from tqdm import tqdm
# import pandas as pd
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchsummary import summary
from skimage.transform import resize
from BinaryMergerDataset import BinaryMergerDataset, get_transforms

# In[4]:


path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
#stretch = AsinhStretch()
pad_val = 0
BATCH_SIZE = 64
NUM_EPOCHS = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# # In[5]:


# torch.cuda.is_available()


# # In[6]:


# class BinaryMergerDataset(Dataset): #in future: put this in one file and always call it!
#     def __init__(self, data_path, dataset, mergers = True, transform=None, codetest=True):
#         self.dataset = dataset
#         self.mergers = mergers
#         self.codetest=codetest
#         if self.dataset == 'train':
#             if mergers == True:
#                 self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'training/anymergers/mergerlabel.npy')
#                 self.image_filenames = self.images
#                 #print('length of file list', len(self.images))
#             else:
#                 self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'training/nonmergers/mergerlabel.npy')
#                 #print('length of file list', len(self.images))
#         elif self.dataset == 'validation':
#             if mergers == True:
#                 self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'validation/anymergers/mergerlabel.npy')
#             else:
#                 self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'validation/nonmergers/mergerlabel.npy')
#         elif self.dataset == 'test':
#             if mergers == True:
#                 self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
#             else:
#                 self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
#                 self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
        
#         self.transform = transform
        

#     def __len__(self):
#         if self.codetest:
#             return len(self.img_labels[0:200])
#         else:   
#             return len(self.img_labels)
    
#     def __getitem__(self, idx):
#         #print(np.shape(self.images))
#         img_path = self.images[idx]
#         #print(idx)
#         image = np.load(img_path) #keep as np array to normalize
#         #print('first shape', np.shape(image))
#         #image = np.transpose(image, (2,1,0))
#         #print('image shape', np.shape(image))
#         image1 = image[:,:,0:3]
#         #print('necesary image shape: ', np.shape(image1))
#         image = image[:, :, [0, 1, 3]]
#         #image = image[:,:,[[0,2,3]]][:,:,0,:] #had to play around with indices to get rid of 606 filter
#         #print('image shape: ', np.shape(image))
#         #image = stretch(image)
#         ##clip values to get rid of outliers -- np.clip makes the percentiles the new min and max
#         #look in Filter.ipynb for details
#         global_min = -8.862120426258897e-20 
#         global_max = 2.373746629900744e-16
#         image = np.clip(image, a_min=global_min, a_max=global_max)
#         #resize to 224 for resnet
#         image = resize(image, (224,224))
#         #print('image shape: ', np.shape(image))
#         image = image * 1e20 #test to get magnitudes up
#         # power = simple_norm(image, 'power', power = 0.5)
#         # image = power(image)
#         norm = simple_norm(image, 'log', log_a=2e5)
#         image = norm(image)
#         label_file = self.img_labels
#         #print('label shape', np.shape(label_file))
#         label = label_file[idx]
#         #print('first label call: ', np.shape(label))
#         # if label != 0:
#         #     print(label)
#         #print(labels)
#         #label = np.load(label_path)[idx]
#         #print('label shape: ',np.shape(labels))
#         if self.transform is not None:
#             image = self.transform(image)

#         return image, int(label)


# # In[7]:


# def get_transforms(aug=True):
#     transforms = []
#     transforms.append(T.ToTensor())
#     #transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     if aug == True:
#         transforms.append(torch.nn.Sequential(
#         T.RandomRotation(30), 
#         T.RandomHorizontalFlip(0.5),
#         T.RandomVerticalFlip(0.5),
#         T.Pad(pad_val), #this line added for resnet
#         ))
#     else: transforms.append(T.Pad(pad_val))
        
#     return T.Compose(transforms)

# In[8]:


train_mergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=False)
train_nonmergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=False)

train_mergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=False), codetest=False)
train_nonmergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=False), codetest=False)

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset_augment, train_nonmergers_dataset_augment, train_mergers_dataset_orig, train_nonmergers_dataset_orig])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE)

# validation_mergers_dataset_augment = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(aug=True), codetest=False)
# validation_nonmergers_dataset_augment = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(aug=True), codetest=False)

validation_mergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(aug=False), codetest=False)
validation_nonmergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(aug=False), codetest=False)

validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset_orig, validation_nonmergers_dataset_orig])
validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE)#num workers used to be 4


# In[9]:


total_size_train = len(train_dataset_full)
total_size_validation = len(validation_dataset_full)

print(total_size_train)

number_of_iterations = total_size_train//BATCH_SIZE #// is floor
print('number_of_iterations', number_of_iterations)

###https://www.kaggle.com/code/kvpratama/pretrained-resnet18-in-pytorch
model = models.resnet18(weights='DEFAULT') #next instead of pretrained try weights='DEFAULT'
print(model)


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
set_parameter_requires_grad(model)
# In[12]:# 
# Initialize new output layer
model.fc = nn.Linear(512, 1)
model = model.to(device)
#Check which layer in the model that will compute the gradient
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)


modelloss = {} #loss history
modelloss['train'] = []
modelloss['validation'] = []
modelacc = {} #used later for accuracy
modelacc['train'] = []
modelacc['validation'] = []
x_epoch = []

columndata = {}
columndata['accuracy_column'] = []
columndata['output_column'] = []
columndata['label_column'] = []


# In[13]:


def get_accuracy(pred,original):
    pred = (torch.sigmoid(pred) > 0.5).float() #pred is now 0 or 1
    pred = pred.cpu().detach().numpy()
    original = original.cpu().numpy()
    # final_pred= []

    # for i in range(len(pred)):
    #     if pred[i] <= 0.5:
    #         final_pred.append(0)
    #     if pred[i] > 0.5:
    #         final_pred.append(1)
    # final_pred = np.array(final_pred)
    # count = 0

    # for i in range(len(original)):
    #     if final_pred[i] == original[i]:
    #         count+=1
    #         columndata['accuracy_column'].append('yes')
    #     else:
    #         columndata['accuracy_column'].append('no')
    # return count/len(final_pred)*100
    return np.mean(pred == original) * 100

def plot_confusion_matrix(cm, classes, epoch): #help from chat GPT
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix After' + str(epoch) + 'Epochs')
    plt.savefig('ConfusionMatrix_ResNet18_Adam' + str(learning_rate) + '.png', dpi = 300)

# In[14]:
learning_rate =  0.0001 #0.03222646114865107
max_learning_rate = learning_rate*10
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
step_size = number_of_iterations*2 #taking out *NUM_EPOCHS 
##experiments show that it often is good to set stepsize equal to 2 − 10 times the number of iterations in an epoch. - Smith 2017
#step size is during how many epochs will the LR go up from the lower bound, up to the upper bound.
print('step size', step_size)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr = learning_rate, max_lr = max_learning_rate, cycle_momentum=False, step_size_up=step_size)

# targets_GradCAM = [model.relu1, model.relu3, model.conv3]
# layer_names_GradCAM = ['relu1', 'relu3', 'conv3']


fig = plt.figure(figsize = (12, 6))
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Accuracy")
plt.suptitle('lr = ' + str(learning_rate) + '| cyclic max = ' + str(max_learning_rate) + 'step size =' + str(step_size) + ' batch' + str(BATCH_SIZE))

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    # print('xshape', np.shape(x_epoch))
    # print('yshape', np.shape(modelloss['train']))
    ax0.plot(x_epoch, modelloss['train'], 'b', label='train')
    ax0.plot(x_epoch, modelloss['validation'], 'r', label='val')
    ax1.plot(x_epoch, modelacc['train'], 'b', label='train')
    ax1.plot(x_epoch, modelacc['validation'], 'r', label='val')
    ax1.axhline(y= 65, color = 'grey', linestyle = '--')
    ax1.axhline(y= 75, color = 'grey', linestyle = '--')
    if np.max(modelloss['validation'])> 5:
        ax0.set_ylim(0, 5)
    else:
        ax0.set_ylim(0,np.max(modelloss['validation']) + 0.5)
    ax1.set_ylim(0,100)
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    plt.tight_layout()
    fig.savefig('metrics_ResNet18_Adam_scheduled' + str(learning_rate) + '.png')

# In[15]:

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float(np.inf)
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        # If this is the first validation loss value or if it's better than the best we've seen
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:  # No improvement
            self.counter += 1
            if self.counter >= self.patience:  # If we've reached the patience, stop early
                self.early_stop = True
                print('Early Stopped!')

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
earlystop = EarlyStopping(patience=5, delta = 0.001)        
for epoch in range(NUM_EPOCHS):
    t_epoch_loss = 0.0
    v_epoch_loss = 0.0
    t_accuracy = []
    v_accuracy = []
    t_counter = 0
    v_counter = 0
    #train_error_count = 0.0
    for images, labels, names in tqdm(iter(train_dataloader)):
        model.train(True) #default is not in training mode - need to tell pytorch to train
        bs = images.shape[0]         #batch size
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        #print(labels)
        #print(images.shape[0])
        #print(images.size())
        # print(type(images))
        # print(type(images.double()))
        # print(type(images))
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        pred = (probabilities > 0.5).float()  # Pred is now 0 or 1
        #outputs, features = model(images)
        for o in range(len(outputs)):
            columndata['output_column'].append(pred[o].item())
            columndata['label_column'].append(labels[o].item())
        # print('np shape output: ', np.shape(outputs))
        # print('.shape output: ', outputs.shape)
        # print('outputs',outputs)
        # Apply sigmoid to get probabilities
        labels = labels.double()
        labels = labels.unsqueeze(1)
        pred = pred.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        loss = F.binary_cross_entropy(probabilities, labels)
        optimizer.zero_grad()
        #print(outputs.size())
        #print(labels.size())
        loss.backward()
        optimizer.step()
        t_epoch_loss += loss.item() * bs #make the loss a number
        t_accuracy.append(get_accuracy(pred, labels))
        t_counter +=1 
    #add learning rate scheduling to decrease learning rate as we train
    #before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    # after_lr = optimizer.param_groups[0]["lr"]
    # print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
 
    #look at outputs here and what shape I want!   
    modelacc['train'].append(np.mean(np.array(t_accuracy)))

    #print('loss shape', np.shape(loss))
    modelloss['train'].append(t_epoch_loss/total_size_train)

    print('TRAINING LOSS: ', t_epoch_loss)

    model.eval()
    all_labels = []
    all_preds = []
    all_features = [] #for t-SNE
    saliency_maps = []
    #with torch.no_grad():
    for images, labels, names in tqdm(iter(validation_dataloader)):
        #model.train(False)
        bs = images.shape[0] 
        #model.eval() #added 5/13/23
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        # print('batch size', images.shape[0])
        # print('image shape', np.shape(images))
        # #print(images[100:100,100:100])
        # print('labels', labels)
        
        # Set requires_grad to True for saliency map calculation ##help from chatGPT
        images.requires_grad_()

        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        pred = (probabilities > 0.5).float()  # Pred is now 0 or 1
        # print('output shape', np.shape(outputs))
        # print('outputs', outputs)
        maxvals, preds = torch.max(outputs, 1) #help from chat GPT maxvals sort which class it belongs to but we only want preds
        all_labels.extend(labels.cpu().numpy())
        #all_preds.extend(preds.cpu().numpy())
        #outputs = outputs.detach().numpy()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_features.extend(pred.detach().cpu().numpy())
        labels = labels#.double()
        labels = labels.unsqueeze(1)
        pred = pred.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        loss = F.binary_cross_entropy(probabilities, labels)
        v_epoch_loss += loss.item() * bs
        v_accuracy.append(get_accuracy(pred, labels))
        v_counter += 1
        
        model.zero_grad()
        #outputs[0, all_preds[0]].backward()  # Compute gradients w.r.t the predicted class
        gradients_array = torch.zeros_like(outputs)
        gradients_array[:, 0] = 1  # Assuming class 0 is at index 0
        outputs.backward(gradient=gradients_array) 

        # #chatGPT saliency map help 
        # model.zero_grad()
        # #outputs[0, all_preds[0]].backward()  # Compute gradients w.r.t the predicted class
        # gradients_array = torch.zeros_like(outputs)
        # gradients_array[:, 0] = 1  # Assuming class 0 is at index 0
        # outputs.backward(gradient=gradients_array) 
        # saliency = images.grad.data.abs().squeeze().cpu().numpy()  # Get the absolute gradients
        # saliency_maps.append(saliency)
        
    modelacc['validation'].append(np.mean(np.array(v_accuracy)))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_features = np.array(all_features)
    #print('all_preds1', all_preds)
    all_preds = np.rint(all_preds)

    modelloss['validation'].append(v_epoch_loss/total_size_validation)
    #modelerr['validation'].append(1.0 - validation_epoch_accuracy) 
    earlystop(val_loss=v_epoch_loss/total_size_validation, model=model)
    draw_curve(epoch)
    
##save for reloading and plots
#torch.save(model.state_dict(), 'model_weights.pth')
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'loss': loss,
            }, 'ResNet18CNN.pth')

#CONFUSION MATRIX
print('allfeatures', np.shape(all_features))
print('alllabels', np.shape(all_labels))
cm = confusion_matrix(all_labels, all_features)
plot_confusion_matrix(cm, ['Merger', 'Nonmerger'], epoch)


#t-SNE plot!
#print('shape preds', np.shape(all_features))
# perplexity = [5, 10, 15, 20, 30, 50, 70, 100] #changing this really changes results
# colors_tSNE = ['green' if label == 1 else 'purple' for label in all_labels] 
# for p in perplexity:
#     tSNE = TSNE(n_components = 2, perplexity= p, learning_rate="auto")
#     tSNE_transforms = tSNE.fit_transform(all_features)
#     plt.figure()
#     plt.scatter(tSNE_transforms[:, 0], tSNE_transforms[:, 1], c = colors_tSNE)
#     plt.title('t-SNE visualization | perplexity = ' + str(p))
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#     plt.savefig('tsne/tSNE_perplexity' + str(p) + '.png')


#### GradCAM with help from github and chat GPT
# We have to specify the target we want to generate the CAM for.
# targets = [ClassifierOutputTarget(0)] #target class is mergers
# # Select the first validation image to visualize Grad-CAM
# sample_index_choices = np.arange(0,10)  # Change this index to visualize different images
# for sample_index in sample_index_choices:
#     sample_image = validation_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
#     sample_label = validation_dataloader.dataset[sample_index][1]

#     # Move to device
#     sample_image = sample_image.to(torch.float32).to(device)
#     print(type(sample_image))
    
#     # Get the original image for visualization
#     original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
#     predicted_class = model(sample_image).argmax(dim=1).item()
#     predicted_label = f"Predicted: {predicted_class}, True: {sample_label}"


#     # Generate the Grad-CAM visualization
#     #targets = [ClassifierOutputTarget(int(sample_label))]  # Use the true label as target
#     counter = 0
#     for target_layer_GradCAM in targets_GradCAM:    
#         layername = layer_names_GradCAM[counter]
#         counter += 1
#         print('target', target_layer_GradCAM)
#         with GradCAM(model=model, target_layers=[target_layer_GradCAM]) as cam:
#             print('shapes', np.shape(sample_image), np.shape(targets))
#             grayscale_cam = cam(input_tensor=sample_image, targets=targets)
#             grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch

#         # Convert the input image to a format suitable for visualization
#         rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
#         grayscale_cam = grayscale_cam#.numpy()

#         # Visualize the Grad-CAM output
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#         # Plot the Grad-CAM
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
#         # Original image subplot
#         im1 = ax1.imshow(original_img) #assign image for colorbar
#         ax1.set_title(f"Original Image - (True: {sample_label})")
#         ax1.axis('off')  # Turn off axis
#         cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', norm=LogNorm(vmin=np.min(original_img), vmax=np.max(original_img)))
#         cbar1.set_label('Intensity (log scale)')
#         #GradCAM
#         im2 = ax2.imshow(visualization)
#         ax2.title(f'Grad-CAM for Validation Image {sample_index} Layer {layername} (True Label: {sample_label})')
#         ax2.axis('off')
#         cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
#         cbar2.set_label('Grad-CAM Intensity')
        
#         plt.tight_layout()
#         plt.savefig(f'gradcam/gradcam_validation_image_{layername}_{sample_index}.png', dpi=300)
#         plt.clf()
#         plt.close()
            


#     # fig = plt.figure()
#     # ax0 = fig.add_subplot(121, title="Loss")
#     # ax1 = fig.add_subplot(122, title="Accuracy")
#     # plt.suptitle('lr = ' + str(learning_rate))
#     # ax0.plot(np.arange(NUM_EPOCHS), modelloss['train'], 'b', label='train')
#     # ax0.plot(np.arange(NUM_EPOCHS), modelloss['validation'], 'r', label='val')
#     # ax1.plot(np.arange(NUM_EPOCHS), modelacc['train'], 'b', label='train')
#     # ax1.plot(np.arange(NUM_EPOCHS), modelacc['validation'], 'r', label='val')
#     # #ax0.set_ylim(0, 5)
#     # ax1.set_ylim(0,100)
#     # plt.tight_layout()
#     # fig.savefig('metrics_Notebook_Adam_scheduled' + str(learning_rate) + '.png')






