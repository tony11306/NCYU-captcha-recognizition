import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms


from model import Model
from dataset import Dataset

# train captcha recognition model
TRAINING_DATA_DIR = 'NCYU_Captcha_Dataset/trainset/'
TEST_DATA_DIR = 'NCYU_Captcha_Dataset/testset/'
def rpt(x):
    return x.repeat(1, 1, 1)

def accuracy(output, target):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('initializing model...')
    model = Model()
    model.to(device)
    dataset = Dataset(TRAINING_DATA_DIR, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(rpt)
        ]))
    
    test_dataset = Dataset(TEST_DATA_DIR, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(rpt)
        ]))

    
    
    print('loading data...')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)
    print('loading data done')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('start training...')
    # 迭代訓練模型
    for epoch in range(70):
        loss_history = []
        acc_history = []
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, labels)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))

        #if torch.mean(torch.Tensor(acc_history)) > 0.99:
        #    break
        loss_history = []
        acc_history = []
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels)
                acc_history.append(float(acc))
                loss_history.append(float(loss))
            print('test_loss: {:.4}|test_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        


    # save model
    torch.save(model, 'model.pth')
    print('Finished Training')