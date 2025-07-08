import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

'''
Basic Dataset Class:
    Used as prep for the PyTorch data loader
'''
class ImageDataset(Dataset):
    # Constructor method
    def __init__(self, data_dir, val_split, transform=None):
        # Directory containing the images
        self.data_dir = data_dir
        # Transforms for __getitem__
        self.transform = transform
        # Lists to store all images, and divide them by class
        self.image_files = []
        self.pos_class = []
        self.neg_class = []
        # Go through the directory and read any pictures within
        for file in os.listdir(data_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                self.image_files.append(file)
                # Filenames starting with 'a' are positive
                if file[0].lower() == 'a':
                    self.pos_class.append(file)
                # Filenames starting with 'f' are negative
                elif file[0].lower() == 'f':
                    self.neg_class.append(file)
        # Used with random_split from PyTorch during testing
        if val_split > 0.0 :
            self.val_size = val_split
            self.train_size = 1.0 - val_split

    # Returns the number of data images
    def __len__(self):
        return len(self.image_files)

    # Used by PyTorch to create data loaders, returns a image and its label
    def __getitem__(self, index):
        # Find image and open it
        filename = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(filename).convert('RGB')
        # Default negative label
        label = 0
        # Positive label if filename reflects such
        if filename[0].lower() == 'a':
            label = 1
        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label

# Data Loader Method: initialized both training and validation data loaders
def trainValidDataLoader(data_dir, batch_size=32, val_split=0.2, transform=None):
    # Get both datasets from their directories
    train_dataset = ImageDataset(data_dir + '/Train', val_split, transform=transform)
    val_dataset = ImageDataset(data_dir + '/Validation', val_split, transform=transform)
    # Turn both into proper data loaders so they can be trained on with PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

#Validate using a dataloader
def validateD(valLoader, model, device='cpu'):
    #switch to evaluate mode
    model.eval()
    acc = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

# Train Method: teaches model for use on validation set
def trainModel(model, train_loader, val_loader, epochs=100, lr=0.0001, device='cpu'):
    # Move model to desired device
    model.to(device)
    # Set loss function to CEL, best for classification
    loss_function = nn.CrossEntropyLoss()
    # Set optimizer to Adam to update models parameters based on gradients
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Main training loop
    for epoch in range(epochs):
        # Set model to train mode
        model.train()
        # Initialize running loss to keep track of the loss across all batches
        running_loss = 0.0
        # Inner loop: process each batch of images from training loader
        for images, labels in train_loader:
            # Move input images and their labels to same device as the model
            images = images.to(device)
            labels = labels.to(device)
            # Clear gradients from past batch, they accumulate by default
            optimizer.zero_grad()
            # Complete a forward pass
            outputs = model(images)
            # Calculate loss based on incorrect predictions
            loss = loss_function(outputs, labels)
            # Compute gradients
            loss.backward()
            # Update parameters using gradients
            optimizer.step()
            # Accumulate running loss
            running_loss += loss.item()

        # Compute average loss from all batches in this epoch
        avg_loss = running_loss / len(train_loader)
        acc = validateD(val_loader, model, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        print(f"Validation for epoch [{epoch+1}/{epochs}], = {acc * 100:.2f}%")


