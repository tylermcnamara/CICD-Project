import torch
from torchvision import transforms
from model import LeNet
from utils import trainValidDataLoader, train_model, evaluate_model

if __name__ == "__main__":
    # Set up transform
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])

    # Load data
    train_loader, val_loader = trainValidDataLoader(
        data_dir="Dataset",
        batch_size=4,
        transform=transform
    )

    # Initialize model
    model = LeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)

    # Evaluate after training (optional if not already included)
    evaluate_model(model, val_loader, device)
