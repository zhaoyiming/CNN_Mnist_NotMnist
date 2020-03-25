import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import inputminst
def data():
    X, Y = inputminst.get_file('./notMNIST_small')
    X_train = X[:2000]
    X_train = X_train.reshape(-1, 1, 28, 28)
    Y_train = Y[:2000]

    X_test = X[-200:-1]
    X_test = X_test.reshape(-1, 1, 28, 28)
    Y_test = Y[-200:-1]

    Batch_size=64
    train_dataset=Data.TensorDataset(torch.tensor(X_train).float(),torch.tensor(Y_train).float())
    train_loader=Data.DataLoader(
        dataset=train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=2
    )
    test_dataset = Data.TensorDataset(torch.tensor(X_test).float(),torch.tensor(Y_test).float())
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=2
    )
    return train_loader,test_loader
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(1,6,5,1,2)
        self.conv2=nn.Conv2d(6,16,5)

        self.fc1=nn.Linear(16*5*5,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))

        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features
net=Net()

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)
trainloader,testloader=data()

for epoch in range(40):
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        optimizer.zero_grad()
        output=net(inputs)
        loss=criterion(output,torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:

            print('[%d, %5d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0





correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

print('Accuracy of the network on the 200 test images: %d %%' % (100 * correct / total))

