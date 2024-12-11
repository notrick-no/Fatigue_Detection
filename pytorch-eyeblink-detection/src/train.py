import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
import model
import csv
import matplotlib.pyplot as plt
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

# 定义Summary_Writer
writer = SummaryWriter('../runs/exp2')   # 数据存放在这个文件夹


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if not torch.cuda.is_available():
    from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #

shape = (48, 48)

validation_ratio = 0.1


def resize(image, bbox):
    (x,y,w,h) = bbox
    eye = image[y:y + h, x:x + w]
    return Image.fromarray(cv2.resize(eye, shape))



class DataSetFactory:

    def __init__(self):
        images = []
        labels = []

        files = list(map(lambda x: {'file': x, 'label':1}, glob.glob('../dataset/eye_images/open/*.jpg')))
        files.extend(list(map(lambda x: {'file': x, 'label':0}, glob.glob('../dataset/eye_images/close/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':0}, glob.glob('../dataset/eye_images/closedLeft/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':0}, glob.glob('../dataset/eye_images/closedRight/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':1}, glob.glob('../dataset/eye_images//openLeft/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':1}, glob.glob('../dataset/eye_images//openRight/*.jpg'))))
        random.shuffle(files)
        for file in files:
            img = cv2.imread(file['file'])
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(file['label'])

        validation_length = int(len(images) * validation_ratio)
        validation_images = images[:validation_length]
        validation_labels = labels[:validation_length]
        images = images[validation_length:]
        labels = labels[validation_length:]
        self.train_num = len(images)
        self.val_num   = len(validation_images)

        print('training size %d : val size %d' % (len(images), len(validation_images)))

        train_transform = transforms.Compose([
            ToTensor(),
        ])
        val_transform = transforms.Compose([
            ToTensor(),
        ])

        self.training = DataSet(transform=train_transform, images=images, labels=labels)
        self.validation = DataSet(transform=val_transform, images=validation_images, labels=validation_labels)


class DataSet(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, labels=None):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


def main():
    # variables  -------------
    batch_size = 64
    lr = 0.001
    epochs = 300
    # ------------------------
    print(device)
    factory = DataSetFactory()
    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(factory.validation, batch_size=batch_size, shuffle=True, num_workers=1)
    network = model.Model(num_classes=2).to(device)
    if not torch.cuda.is_available():
        summary(network, (1, shape[0], shape[1]))

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)  #优化器
    criterion = nn.CrossEntropyLoss()

    min_validation_loss = 10000
    running_loss = 0.0
    for epoch in range(epochs):
        network.train()  #启用 Batch Normalization 和 Dropout  network.eval()则不启用
        total = 0
        correct = 0
        total_train_loss = 0
        train_loss = 0.0

        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = network(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data
            running_loss += loss.item()
            train_loss += loss.item()
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()
        
            if i % 1000 == 999:    # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss', running_loss / 1000, epoch * len(training_loader) + i)             
                            
                running_loss = 0.0
        train_accuracy = 100. * float(correct) / total            
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), train_accuracy))
        writer.add_scalar('train loss', total_train_loss / (i + 1), epoch )
        writer.add_scalar('train accuracy', train_accuracy, epoch )

        network.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0
            validation_loss = 0
            for j, (x_val, y_val) in enumerate(validation_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = network(x_val)
                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

            validation_accuracy = 100. * float(correct) / total
            if total_validation_loss <= min_validation_loss:
                if epoch >= 3:
                    print('saving new model')
                    state = {'net': network.state_dict()}
                    torch.save(state, '../trained/model_%d_%d_%.4f.t7' % (epoch + 1, validation_accuracy, total_validation_loss / (j + 1)))
                min_validation_loss = total_validation_loss

            print('Epoch [%d/%d] validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), validation_accuracy))
            writer.add_scalar('eval loss', total_validation_loss / (j + 1), epoch )
            writer.add_scalar('eval accuracy', validation_accuracy, epoch )
            
    print('\nTrain done!')
    print('Epoch:%d, batch_size:%d, Iter:%d, training size:%d, val size:%d' % (epochs, batch_size, (epoch * len(training_loader) + i), factory.train_num, factory.val_num))
    print('Training Loss: %.4f, Training Accuracy: %.4f, Validation Loss: %.4f, Validation Accuracy: %.4f'
            %(total_train_loss / (i + 1), train_accuracy, total_validation_loss / (j + 1), validation_accuracy))
    

if __name__ == "__main__":
    main()
