#This is the version of DataManagerPytorch(L) for lectures in spring 2025
#Current version: 1.0 created on Jan 1, 2025. Please do not remove these comments.
import torch
import utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import math
import random

#Class to help with converting between dataloader and pytorch tensor
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader
def validateD(valLoader, model, device=None):
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
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
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

#Replicate TF's predict method behavior
#Returns class labels predicted by models on a dataloader
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset)
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index
    return xData, yData

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#This data is in the range 0 to 1
def GetCIFAR10Validation(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Training(imgSize = 32, batchSize=128):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses)
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yPred = predictD(dataLoader, numClasses, model, device)
    for i in range(0, xData.shape[0]): #Go through every sample
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0)
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    '''
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    '''
    #Assume we have enough samples now, restore in a properly shaped array
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Show 20 images, 10 in first and row and 10 in second row
def ShowImages(xFirst, xSecond):
    #Convert from Pytorch tensor to HxWxColorChannel and Numpy
    xFirstNumpy = xFirst.detach().numpy().squeeze(1)
    xSecondNumpy = xSecond.detach().numpy().squeeze(1)
    #xFirstNumpy = torch.transpose(xFirst,0,2,3,1).numpy()
    #xSecondNumpy = torch.transpose(xSecond,0,2,3,1).numpy()
    n = 6  # how many digits we will display
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirstNumpy[i], cmap='gray')
        ax.axis('off')
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecondNumpy[i], cmap='gray')
        ax.axis('off')
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
    plt.show()