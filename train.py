import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random
from tqdm import tqdm
import json

from utils import set_seed, plot_stats
from model import Model
from IPython.display import clear_output

set_seed(42)


def parse_args():
    """Function for parsing arguments"""
    parser = argparse.ArgumentParser(description="Training script for CIFAR-10 classification")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


# train model
def train(model, train_loader, device, optimizer, loss_fn) -> float:
    """Function for training"""
    model.train()

    train_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy


# evaluate results
@torch.inference_mode()
def evaluate(model, loader, device, loss_fn) -> tuple[float, float]:
    """Function for evaluating a model"""
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device), y.to(device)

        output = model(x)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy


def whole_train_valid_cycle_with_scheduler(model, scheduler, optimizer, loss_fn, train_loader, valid_loader, device, num_epochs, title):
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, device, optimizer, loss_fn)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, device, loss_fn)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        clear_output()

        plot_stats(
            train_loss_history, valid_loss_history,
            train_accuracy_history, valid_accuracy_history,
            title
        )

        scheduler.step()


def main():
    """Main function for launching testing"""
    args = parse_args()
    print(f"Training CIFAR-10 for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.lr}")

    # Устройство
    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset_train = CIFAR10(root='data', train=True, download=True, transform=T.ToTensor())

    # calculate the mean and standard deviation to normalize the data
    means = (dataset_train.data / 255).mean(axis=(0, 1, 2))
    stds = (dataset_train.data / 255).std(axis=(0, 1, 2))

    print("means: ", means)
    print("stds: ", stds)

    # save data for normalization
    normalize_dict = {
        "means": means,
        "stds": stds
    }
    # сonvert to list
    normalize_dict = {
        'means': normalize_dict['means'].tolist(),
        'stds': normalize_dict['stds'].tolist()
    }
    # save json file
    with open("normalize_data.json", "w", encoding="utf-8") as json_file:
        json.dump(normalize_dict, json_file, indent=4)

    train_transforms = T.Compose(
        [
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
    )

    test_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
    )

    train_dataset = CIFAR10(root='data', train=True, transform=train_transforms)
    valid_dataset = CIFAR10(root='data', train=False, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Model().to(device)
    # define loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20)

    whole_train_valid_cycle_with_scheduler(model=model,
                                           scheduler=scheduler,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           train_loader=train_loader,
                                           valid_loader=valid_loader,
                                           device=device,
                                           num_epochs=args.epochs,
                                           title='Training the model')

    # Сохранение модели
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as cifar10_model.pth")


if __name__ == "__main__":
    main()

