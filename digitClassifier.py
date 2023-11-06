import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
        transforms.ToTensor(),
    ])

mnist_train = torchvision.datasets.MNIST(root="./MNIST", train=True, download=True, transform=transform);
mnist_test = torchvision.datasets.MNIST(root="./MNIST", train=False, download=True, transform=transform);
mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self,x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.gelu(self.fc5(x))
        x = self.fc6(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-2)
num_epochs = 15

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(mnist_train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Network Loss: " + str(loss.item()))

print("Finished Training")

net.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for data in mnist_test_loader:
        images, labels = data
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

networks_dir = 'Networks'
os.makedirs(networks_dir, exist_ok=True)  # Create the directory if it doesn't exist
model_path = os.path.join(networks_dir, f'net_{accuracy:.2f}.pth')

torch.save(net.state_dict(), model_path)
print(f'Saved trained model with accuracy {accuracy:.2f}% as {model_path}')
