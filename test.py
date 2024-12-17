import json
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from model import Model


def parse_args():
    """Function for parsing arguments"""
    parser = argparse.ArgumentParser(description="Test CIFAR-10 model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--checkpoint", type=str, default="cifar10_model.pth", help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for testing (cuda or cpu)")
    return parser.parse_args()


def test(model, test_loader, criterion, device):
    """Function for testing """
    model.eval()  # put the model into testing mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradients for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # predictions
            loss = criterion(outputs, labels)  # calculate loss
            running_loss += loss.item()

            _, predicted = outputs.max(1)  # get predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    """Main function for launching testing"""
    args = parse_args()
    print(f"Testing model from checkpoint: {args.checkpoint}")
    print(f"Using device: {args.device}")

    # define a device
    device = torch.device(args.device)

    file_path = "normalize_data.json"

    # download  JSON with data for normalization
    with open(file_path, "r", encoding="utf-8") as json_file:
        loaded_dict = json.load(json_file)

    # convert lists to np.array
    normalize_dict = {
        'means': np.array(loaded_dict['means']),
        'stds': np.array(loaded_dict['stds'])
    }

    # prepare data
    test_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=normalize_dict['means'], std=normalize_dict['stds'])
        ]
    )

    test_dataset = CIFAR10(root="./data", train=False, transform=test_transforms, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # download the model
    model = Model().to(device)  # Создаем экземпляр модели
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))  # download weights
    print("Model loaded successfully!")

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # test the model
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()