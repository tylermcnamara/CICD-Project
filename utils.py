import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, data_dir, val_split, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        self.image_files = []
        self.pos_class = []
        self.neg_class = []

        for file in os.listdir(data_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                self.image_files.append(file)
                if file[0].lower() == 'a':
                    self.pos_class.append(file)
                elif file[0].lower() == 'f':
                    self.neg_class.append(file)

        if val_split > 0.0 :
            self.val_size = val_split
            self.train_size = 1.0 - val_split


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(filename).convert('RGB')

        label = 0

        if filename[0].lower() == 'a':
            label = 1

        if self.transform:
            image = self.transform(image)

        return image, label


def trainValidDataLoader(data_dir, batch_size=32, val_split=0.2, transform=None):
    train_dataset = ImageDataset(data_dir + '/Train', val_split, transform=transform)
    val_dataset = ImageDataset(data_dir + '/Validation', val_split, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def evaluate_model(model, loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')



def train_model(model, train_loader, val_loader, epochs=100, lr=0.0001, device='cpu'):
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        evaluate_model(model, val_loader, device)






