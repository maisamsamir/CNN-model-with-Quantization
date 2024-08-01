import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import brevitas.nn as qnn
import brevitas.core as bnn
import brevitas.nn as qnn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class QuantizedCNN(nn.Module):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(1, 32, kernel_size=3, padding=1, weight_bit_width=8, bias=False),
            qnn.QuantReLU(),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2)
        )
    
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(32, 64, kernel_size=3, weight_bit_width=8, bias=False),
            qnn.QuantReLU(),
            qnn.QuantMaxPool2d(kernel_size=2)
        )
        self.fc1 = qnn.QuantLinear(64*6*6, 1000, weight_bit_width=8, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = qnn.QuantLinear(1000, 10, weight_bit_width=8, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = QuantizedCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')

conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=train_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:5]))
print('GroundTruth: ', ' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(5)))

model.eval()
outputs = model(images[:5])
_, predicted = torch.max(outputs, 1)

class_names = train_dataset.classes
predicted_labels = [class_names[predicted[j].item()] for j in range(5)]

print('Predicted: ', ' '.join('%5s' % label for label in predicted_labels))

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = transform(image)
    image = image.unsqueeze(0)
    return image

image_path = 'trouser123.jpg'
own_image = load_image(image_path)
own_output = model(own_image)
_, own_predicted = torch.max(own_output, 1)

own_predicted_label = train_dataset.classes[own_predicted.item()]

print('Predicted for custom image: ', own_predicted_label)

own_image = own_image.squeeze(0)
imshow(own_image)