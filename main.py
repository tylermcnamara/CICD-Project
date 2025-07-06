import os
import torch
import DataManagerPytorchL as DMP
import attacks
from torchvision import transforms
from model import LeNet
from utils import trainValidDataLoader, train_model, validateD

def main():
    # Set up transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])
    # Load data
    train_loader, val_loader = trainValidDataLoader(
        data_dir="Dataset",
        batch_size=25,
        transform=transform
    )
    # Initialize model
    model = LeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check for saved weights
    if os.path.exists("lenet_weights.pth"):
        # Saved weights loaded into model
        model.load_state_dict(torch.load("lenet_weights.pth", weights_only=True))
        model.eval()
        print("Model loaded from save")
    else:
        # No saved model, train it then save
        train_model(model, train_loader, val_loader, epochs=10, lr=0.00001, device=device)
        model.eval()
        torch.save(model.state_dict(), "lenet_weights.pth")

    # Evaluate
    acc = validateD(val_loader, model, device)
    print(f"Clean Validation Accuarcy: {acc * 100:.2f}%")
    correctLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(model, 6, val_loader, 2)

    # Do the attacks
    epsilonMax = 0.15  # Maximum perturbation
    clipMin = 0.0  # Minimum value a pixel can take
    clipMax = 1.0  # Maximum value a pixel can take
    numSteps = 10
    epsilonStep = epsilonMax / numSteps # Amount of change per step
    # Load adversarial images
    advLoader = attacks.PGDNativePytorch(device, val_loader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax)
    # Get accuarcy of model on malicious examples
    advAcc = validateD(advLoader, model, device)
    print(f"Adv Accuarcy: {advAcc * 100:.2f}")
    # Clean up images to be shown
    xCleanTensor, yCleaTensor = DMP.DataLoaderToTensor(val_loader)
    xAdvTensor, _ = DMP.DataLoaderToTensor(advLoader)
    DMP.ShowImages(xCleanTensor, xAdvTensor)

if __name__ == "__main__":
   main()
