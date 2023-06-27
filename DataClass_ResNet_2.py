import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import glob
from astropy.visualization import AsinhStretch, simple_norm
#import os
#from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd

path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/'
#stretch = AsinhStretch()
pad_val = int((256-202)/2)
BATCH_SIZE = 32

class BinaryMergerDataset(Dataset): #in future: put this in one file and always call it!
    def __init__(self, data_path, dataset, mergers = True, transform=None, codetest=True):
        self.dataset = dataset
        self.mergers = mergers
        self.codetest=codetest
        if self.dataset == 'train':
            if mergers == True:
                self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/anymergers/mergerlabel.npy')
                #print('length of file list', len(self.images))
            else:
                self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/nonmergers/mergerlabel.npy')
                #print('length of file list', len(self.images))
        elif self.dataset == 'validation':
            if mergers == True:
                self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/nonmergers/mergerlabel.npy')
        elif self.dataset == 'test':
            if mergers == True:
                self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
        
        self.transform = transform
        

    def __len__(self):
        if self.codetest:
            return len(self.img_labels[0:10])
        else:   
            return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        #print(idx)
        image = np.load(img_path) #keep as np array to normalize
        image = image[:,:,1:4]
        #print('image shape: ', np.shape(image))
        #image = stretch(image)
        image = image * 1e20 #test to get magnitudes up
        power = simple_norm(image, 'power', power = 0.1)
        image = power(image)
        label_file = self.img_labels
        #print('first label call: ', np.shape(label))
        label = label_file[idx]
        # if label != 0:
        #     print(label)
        #print(labels)
        #label = np.load(label_path)[idx]
        #print('label shape: ',np.shape(labels))
        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)



# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


def get_transforms(aug=True):
    transforms = []
    transforms.append(T.ToTensor())
    if aug == True:
        transforms.append(torch.nn.Sequential(
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.Pad(pad_val)
        ))
    else: transforms.append(T.Pad(pad_val))
        
    return T.Compose(transforms)

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

accuracylist = []

train_mergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=True)
train_nonmergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=True)

train_mergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=False), codetest=True)
train_nonmergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=False), codetest=True)

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset_augment, train_nonmergers_dataset_augment, train_mergers_dataset_orig, train_nonmergers_dataset_orig])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)

validation_mergers_dataset = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(aug=False), codetest=True)
validation_nonmergers_dataset = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(aug=False), codetest=True)

validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset, validation_nonmergers_dataset])
validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)#num workers used to be 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#common practice is to make a subclass
class ResNet(nn.Module): #inheritance --> can use anything in nn.Module NOT LIKE A FUNCTION
    def __init__(
        self, in_channels: int,  out_channels: int, pretrained: bool = True 
    ):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #self says "this variable belongs to the class"
        #print(self.resnet)
        # Freeze model parameters -- commented out on 5/11/23 to test if that's whats messing with accuracy
        for param in self.resnet.parameters():
            param.requires_grad = False
        #self.resnet.fc = nn.Linear(in_channels, out_channels, bias=True) #bias is like y-intercept #add activation here
        self.resnet.fc = nn.Sequential(torch.nn.Linear(in_channels, out_channels, bias=True), torch.nn.Sigmoid())
        #print(self.resnet)

    def forward(self, x): #how a datum moves through the net
        x = self.resnet(x) #model already has the sequence - propogate x through the network!
        #print(x)
        return x



# examples = next(iter(train_dataloader))

# for label, img  in enumerate(examples):
#     single_img = img[0]
#     img_permuted = single_img.permute(1, 2, 0)
#     # Convert the tensor to a numpy array
#     img_np = img_permuted.numpy()
#     # Plot the image using Matplotlib
#     plt.imshow(img_np)
#     plt.axis('off')  # Optional: turn off the axis labels and ticks
#     plt.show()
   
# exit()
model = ResNet(512, 1, True)
#print(model.forward(train_dataset_full[1]))
model = model.to(device)
model = model.double()
#print(model)

#tweak model
#model.features[0] = torch.nn.Conv2d(model.features[0].kernel_sieze, (5,5))
#model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
#print(model)
NUM_EPOCHS = 10
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0
# training_epoch_loss = []
# val_epoch_loss = []
# training_epoch_accuracy = []
# val_epoch_accuracy = []
# accuracy = []
# train_loss = 0.0
# train_acc = 0.0
# valid_loss = 0.0
# valid_acc = 0.0

modelloss = {} #loss history
modelloss['train'] = []
modelloss['validation'] = []
modelacc = {} #used later for accuracy
modelacc['train'] = []
modelacc['validation'] = []
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Accuracy")

optimizer = optim.Adam(model.parameters(), lr=0.001)

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    print('xshape', np.shape(x_epoch))
    print('yshape', np.shape(modelloss['train']))
    ax0.plot(x_epoch, modelloss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, modelloss['validation'], 'ro-', label='val')
    ax1.plot(x_epoch, modelacc['train'], 'bo-', label='train')
    ax1.plot(x_epoch, modelacc['validation'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('metrics.png')

# trainingloss = 0.0
# correct_labels_train = 0.0
# valloss = 0.0
# correct_labels_val = 0.0
columndata = {}
columndata['accuracy_column'] = []
columndata['output_column'] = []
columndata['label_column'] = []
#from https://neptune.ai/blog/pytorch-loss-functions
def get_accuracy(pred,original):

    pred = pred.detach().numpy()
    original = original.numpy()
    final_pred= []

    for i in range(len(pred)):
        if pred[i] <= 0.5:
            final_pred.append(0)
        if pred[i] >= 0.5:
            final_pred.append(1)
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original)):
        if final_pred[i] == original[i]:
            count+=1
            columndata['accuracy_column'].append('yes')
        else:
            columndata['accuracy_column'].append('no')
    return count/len(final_pred)*100

#task from 6/12: make a table of loss value (per epoch), predicted label, and true label for each image
for epoch in range(NUM_EPOCHS):
    t_epoch_loss = 0.0
    v_epoch_loss = 0.0
    #train_error_count = 0.0
    for images, labels in tqdm(iter(train_dataloader)):
        model.train(True) #default is not in training mode - need to tell pytorch to train
        bs = images.shape[0]         #batch size
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        #print(labels)
        #print(images.shape[0])
        #print(images.size())
        outputs = model(images.double())
        for o in range(len(outputs)):
            columndata['output_column'].append(outputs[o].item())
            columndata['label_column'].append(labels[o].item())
        print('np shape output: ', np.shape(outputs))
        print('.shape output: ', outputs.shape)
        print(outputs)
        labels = labels.double()
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(outputs, labels)
        optimizer.zero_grad()
        #print(outputs.size())
        #print(labels.size())
        loss.backward()
        optimizer.step()
        t_epoch_loss += loss.item() * bs #make the loss a number
#look at outputs here and what shape I want!   
    modelacc['train'].append(get_accuracy(outputs, labels))
    # print('col shapes', np.shape(labels))
    # #print(columndata)
    # print(len(columndata['accuracy_column']), len(columndata['label_column']), len(columndata['output_column']))
    # table = pd.DataFrame.from_dict(columndata)
    # print(table)
    # table.to_csv('LossTestingTableTraining.csv') #currently only works for 1 epoch

    print('loss shape', np.shape(loss))
    modelloss['train'].append(t_epoch_loss)
    #print('loss.item shape', np.shape(loss.item()))
    print(modelloss['train'], np.shape(modelloss['train']))
        # Calculate Loss
    
        
        #trainingloss += loss.item() * BATCH_SIZE
        #correct_labels_train += float(torch.sum(outputs == labels.data))
        #train_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
       # print('length of training loss', len(trainingloss))
    #training_epoch_loss = trainingloss / len(train_dataloader.dataset) 
    #print('TRAINING LOSS: ', type(training_epoch_loss), training_epoch_loss)
    print('TRAINING LOSS: ', t_epoch_loss)
    #training_epoch_accuracy = correct_labels_train / len(train_dataloader.dataset)
    #modelloss['train'].append(training_epoch_loss)
   # modelerr['train'].append(1.0 - training_epoch_accuracy)        
    # training_epoch_loss.append(trainingloss)
    # print('shape of training loss: ', np.shape(training_epoch_loss))
    # train_accuracy = 1.0 - float(train_error_count) / float(len(train_dataset_full))
    # training_epoch_accuracy.append(train_accuracy)
    
    # #val_error_count = 0.0
    model.eval()
    for images, labels in tqdm(iter(validation_dataloader)):
        #model.train(False)
        
        bs = images.shape[0] 
        #model.eval() #added 5/13/23
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        print('batch size', images.shape[0])
        print(np.shape(images))
        print(images[100:100,100:100])
        print(labels)
        outputs = model(images.double())
        print(outputs)
        labels = labels.double()
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(outputs, labels)
        v_epoch_loss += loss.item() * bs
    
    modelacc['validation'].append(get_accuracy(outputs, labels))
        
        #valloss.append(loss.item())
        #val_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        #valloss += loss.item() * BATCH_SIZE
        #correct_labels_val += float(torch.sum(outputs == labels.data))
    #validation_epoch_loss = valloss / len(validation_dataloader.dataset)
    #validation_epoch_accuracy = correct_labels_val / len(validation_dataloader.dataset)
    modelloss['validation'].append(v_epoch_loss)
    #modelerr['validation'].append(1.0 - validation_epoch_accuracy) 
        
    draw_curve(epoch)
    save_checkpoint(model=model, optimizer=optimizer, save_path='/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/savedresnetmodel.txt', epoch = epoch)

    # val_epoch_loss.append(np.array(valloss))
    # val_accuracy = 1.0 - float(val_error_count) / float(len(validation_dataset_full))
    # accuracylist.append(val_accuracy)
    # print('%d: %f' % (epoch, val_accuracy))
    # if val_accuracy > best_accuracy:
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     best_accuracy = val_accuracy

def Accuracy(model, dataloader): #https://blog.paperspace.com/training-validation-and-accuracy-in-pytorch/
    """
    This function computes accuracy
    """
    #  setting model state
    model.eval()

    #  instantiating counters
    total_correct = 0
    total_instances = 0

    #  creating dataloader
    #dataloader = DataLoader(dataset, 64)

    #  iterating through batches
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            #-------------------------------------------------------------------------
            #  making classifications and deriving indices of maximum value via argmax
            #-------------------------------------------------------------------------
            classifications = torch.argmax(model(images), dim=1)

            #--------------------------------------------------
            #  comparing indicies of maximum values and labels
            #--------------------------------------------------
            correct_predictions = sum(classifications==labels).item()

            #------------------------
            #  incrementing counters
            #------------------------
            total_correct+=correct_predictions
            total_instances+=len(images)
    return round(total_correct/total_instances, 3)

#print('best accuracy:', best_accuracy)
#training accuracy
training_accuracy = Accuracy(model, train_dataloader)
validation_accuracy = Accuracy(model, validation_dataloader)

accuracylist = np.array(accuracylist)
np.savetxt('/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/accuracy_resnettransfer.txt', accuracylist)


# ## plot training and validation loss
# plt.figure()
# plt.plot(training_epoch_loss, label = 'training')
# plt.plot(validation_epoch_loss, label = 'validation')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.savefig('ResNet_loss.png')

# plt.figure()
# plt.plot(np.arange(0,NUM_EPOCHS), training_epoch_accuracy, label = 'training')
# plt.plot(np.arange(0,NUM_EPOCHS), accuracylist, label = 'validation')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.savefig('ResNet_accuracy.png')stb