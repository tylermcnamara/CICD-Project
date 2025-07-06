import torch
import DataManagerPytorchL as DMP
#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax)
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#Native (no attack library) implementation of the PGD attack in Pytorch
def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax):
    model.eval()  #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            # Forward pass the data through the model
            output = model(xAdvCurrent)
            # Calculate the loss
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            cost = loss(output, yCurrent)
            cost.backward()
            advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)
            xInit = xData.clone().detach().to(device)
            advTemp = ProjectionOperation(advTemp, xInit, epsilonMax)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader
    return advLoader