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
#import pandas as pd

data_path = 'dogsandcats/'

BATCH_SIZE = 32

class BinaryDataset(Dataset): #in future: put this in one file and always call it!
    def __init__(self, data_path, dataset, dogs = True, transform=None, codetest=True):
        self.dataset = dataset
        self.dogs = dogs
        self.codetest=codetest
        if self.dataset == 'train':
            if dogs == True:
                self.images = glob.glob(data_path + 'training/dog/*.npy')
                self.img_labels = np.load(data_path + 'training/dog/labels.npy')
                #print('length of file list', len(self.images))
            else:
                self.images = glob.glob(data_path + 'training/cat/*.npy')
                self.img_labels = np.load(data_path + 'training/cat/labels.npy')
                #print('length of file list', len(self.images))
        elif self.dataset == 'validation':
            if dogs == True:
                self.images = glob.glob(data_path + 'validation/dog/*.npy')
                self.img_labels = np.load(data_path + 'validation/dog/labels.npy')
            else:
                self.images = glob.glob(data_path + 'validation/cat/*.npy')
                self.img_labels = np.load(data_path + 'validation/cat/labels.npy')
        # elif self.dataset == 'test':
        #     if dog == True:
        #         self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
        #         self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
        #     else:
        #         self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
        #         self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
        
        self.transform = transform
        

    def __len__(self):
        if self.codetest:
            return len(self.img_labels[0:200])
        else:   
            return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        #print(idx)
        image = np.load(img_path) #keep as np array to normalize
        image = image[:,:,0:3] #one image has 4 filters so I want to get rid of that 
        #print('image shape: ', np.shape(image))
        #image = stretch(image)
        # image = image * 1e20 #test to get magnitudes up
        # power = simple_norm(image, 'power', power = 0.1)
        # image = power(image)
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
    transforms.append(torch.nn.Sequential(T.Resize((224, 224)), T.Normalize(
                                           mean=[0.5,0.5,0.5],
                                           std=[0.25, 0.25, 0.25])))
    if aug == True:
        transforms.append(torch.nn.Sequential(
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        ))
    #else: transforms.append(T.Pad(pad_val))
        
    return T.Compose(transforms)

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

accuracylist = []

train_dog_dataset_augment = BinaryDataset(data_path, 'train', dogs = True, transform = get_transforms(aug=True), codetest=False)
train_cat_dataset_augment = BinaryDataset(data_path, 'train', dogs = False, transform = get_transforms(aug=True), codetest=False)

train_dog_dataset_orig = BinaryDataset(data_path, 'train', dogs = True, transform = get_transforms(aug=False), codetest=False)
train_cat_dataset_orig = BinaryDataset(data_path, 'train', dogs = False, transform = get_transforms(aug=False), codetest=False)

train_dataset_full = torch.utils.data.ConcatDataset([train_dog_dataset_augment, train_cat_dataset_augment, train_dog_dataset_orig, train_cat_dataset_orig])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)

validation_dog_dataset_augment = BinaryDataset(data_path, 'validation', dogs = True, transform = get_transforms(aug=True), codetest=False)
validation_cat_dataset_augment = BinaryDataset(data_path, 'validation', dogs = False, transform = get_transforms(aug=True), codetest=False)

validation_dog_dataset_orig = BinaryDataset(data_path, 'validation', dogs = True, transform = get_transforms(aug=False), codetest=False)
validation_cat_dataset_orig = BinaryDataset(data_path, 'validation', dogs = False, transform = get_transforms(aug=False), codetest=False)

validation_dataset_full = torch.utils.data.ConcatDataset([validation_dog_dataset_augment, validation_cat_dataset_augment, validation_dog_dataset_orig, validation_cat_dataset_orig])
validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)#num workers used to be 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#common practice is to make a subclass
class Network(nn.Module): #inheritance --> can use anything in nn.Module NOT LIKE A FUNCTION
    def __init__(self, channels):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size= (5, 5), stride= (2, 2))
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        #self.averagepool1 = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size= (4, 4), stride= (2, 2))
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.averagepool2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size= (3, 3), stride= (2, 2))
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        #self.averagepool3 = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= (3, 3), stride= (2, 2))
        # self.relu4 = nn.ReLU()
        # self.batchnorm4 = nn.BatchNorm2d(num_features=64)
        #self.averagepool4 = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= (3, 3), stride= (2, 2))
        # self.relu5 = nn.ReLU()
        # self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        # self.averagepool5 = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
        self.flatten = nn.Flatten()
        
        #fully connected layers
        self.fc1 = nn.Linear(in_features=5408, out_features=32)
        # self.fc1 = nn.Linear(in_features=128, out_features=64)
        # self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.dropout1 = nn.Dropout2d(p = 0.4)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=16, out_features=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x): #how a datum moves through the net
        print('starting shape', np.shape(x))
        x = self.conv1(x) #propogate x through the network!
        #print('shape after first conv', np.shape(x))
        x = self.relu1(x)
        #print('shape after first relu', np.shape(x))
        #x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        #print('shape after second conv and relu', np.shape(x))
        #x = self.batchnorm2(x)
        x = self.averagepool2(x)
        #print('shape after avg pool', np.shape(x))
        x = self.conv3(x)
        x = self.relu3(x)
        #print('shape after third conv and relu', np.shape(x))
        #x = self.batchnorm3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.batchnorm4(x)
        # x = self.conv5(x)
        # x = self.relu5(x)
        # x = self.batchnorm5(x)
        # x = self.averagepool5(x)
        x = self.flatten(x)
        #print('shape after flatten', np.shape(x))
        x = self.fc1(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.dropout1(x)
        x = self.fc3(x)
        x = self.dropout2(x)
        x = self.relu4(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
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
model = Network(3)
#print(model.forward(train_dataset_full[1]))
model = model.to(device)
model = model.double()
#print(model)

#tweak model
#model.features[0] = torch.nn.Conv2d(model.features[0].kernel_sieze, (5,5))
#model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
#print(model)
NUM_EPOCHS = 50
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

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

fig = plt.figure()
ax0 = fig.add_subplot(211, title="Loss")
ax1 = fig.add_subplot(212, title="Accuracy")
plt.suptitle('lr = ' + str(learning_rate))


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    print('xshape', np.shape(x_epoch))
    print('yshape', np.shape(modelloss['train']))
    ax0.plot(x_epoch, modelloss['train'], 'b', label='train')
    ax0.plot(x_epoch, modelloss['validation'], 'r', label='val')
    ax1.plot(x_epoch, modelacc['train'], 'b', label='train')
    ax1.plot(x_epoch, modelacc['validation'], 'r', label='val')
    ax0.set_ylim(0,25)
    ax1.set_ylim(0,100)
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    plt.tight_layout()
    fig.savefig('DOGandCAT_metrics_simple_lr' + str(learning_rate) + 'moredropout.png')

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

    pred = pred.cpu().detach().numpy()
    original = original.cpu().numpy()
    final_pred= []

    for i in range(len(pred)):
        if pred[i] <= 0.5:
            final_pred.append(0)
        if pred[i] > 0.5:
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
    t_counter = 0
    v_counter = 0
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
        #print('np shape output: ', np.shape(outputs))
        #print('.shape output: ', outputs.shape)
        #print(outputs)
        labels = labels.double()
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(outputs, labels)
        optimizer.zero_grad()
        #print(outputs.size())
        #print(labels.size())
        loss.backward()
        optimizer.step()
        t_epoch_loss += loss.item() * bs #make the loss a number
        t_counter +=1 
#look at outputs here and what shape I want!   
    modelacc['train'].append(get_accuracy(outputs, labels))
    # print('col shapes', np.shape(labels))
    # #print(columndata)
    # print(len(columndata['accuracy_column']), len(columndata['label_column']), len(columndFalseata['output_column']))
    # table = pd.DataFrame.from_dict(columndata)
    # print(table)
    # table.to_csv('LossTestingTableTraining.csv') #currently only works for 1 epoch

    print('loss shape', np.shape(loss))
    modelloss['train'].append(t_epoch_loss/t_counter)
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
        v_counter += 1
    
    modelacc['validation'].append(get_accuracy(outputs, labels))
        
        #valloss.append(loss.item())
        #val_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        #valloss += loss.item() * BATCH_SIZE
        #correct_labels_val += float(torch.sum(outputs == labels.data))
    #validation_epoch_loss = valloss / len(validation_dataloader.dataset)
    #validation_epoch_accuracy = correct_labels_val / len(validation_dataloader.dataset)
    modelloss['validation'].append(v_epoch_loss/v_counter)
    #modelerr['validation'].append(1.0 - validation_epoch_accuracy) 
        
    draw_curve(epoch)
    #save_checkpoint(model=model, optimizer=optimizer, save_path='/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/savedsimplemodel.txt', epoch = epoch)

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
#np.savetxt('/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/accuracy_simple.txt', accuracylist)


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