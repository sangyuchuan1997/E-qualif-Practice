import torch
import torchvision
import torch.utils.data as tud
from models import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def load_data(batch_size=100, num_workers=2):
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    trainloader = tud.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    testloader = tud.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def init_model(model='cnn', device='mps'):
    device = torch.device(device)
    if model == 'cnn':
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if model == 'ResNet':
        model = ResNet50(img_channel=1, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer


def train(model, trainloader, testloader, criterion, optimizer, epochs=20, batch_size=100, device='mps'):
    device = torch.device(device)
    for epoch in range(epochs):
        # train
        model.train()
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))

        # train accuracy
        sum_loss = .0
        sum_correct = 0
        sum_total = 0
        
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        print('train mean loss: {}, accuracy: {}'.format(sum_loss*batch_size /
                                                         len(trainloader.dataset), float(sum_correct / sum_total)))

        # test accuracy
        sum_loss = .0
        sum_correct = 0
        sum_total = 0
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        print('test mean loss: {}, accuracy: {}'.format(sum_loss*batch_size /
                                                        len(testloader.dataset), float(sum_correct / sum_total)))


if __name__ == '__main__':
    trainloader, testloader = load_data()
    model, criterion, optimizer = init_model(model='ResNet')
    train(model, trainloader, testloader, criterion, optimizer)