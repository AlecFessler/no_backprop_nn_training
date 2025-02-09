import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from no_backprop_optim import generate_model_copies, perturb_params, linear_combination_scalars, linear_combine_models

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def load_model(model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        model = MnistModel()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    @staticmethod
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def get_dataloaders(batch_size, num_workers=2):
        mean, std = (0.1307,), (0.3081,)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, test_loader

    @staticmethod
    def train_model(model, train_loader, test_loader, criterion, epochs, device, N=10):
        for epoch in range(epochs):
            losses = [0] * N
            models = generate_model_copies(model, N, device)
            for model in models:
                perturb_params(model, perturb_prob=0.5, perturb_range=0.01)

            with torch.no_grad():
                with tqdm(train_loader, unit='batch') as tepoch:
                    for images, labels in tepoch:
                        images, labels = images.to(device), labels.to(device)
                        for i, model in enumerate(models):
                            model.train()
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            losses[i] += loss.item()

            combination_scalars = linear_combination_scalars(losses)
            model = linear_combine_models(models, combination_scalars, device)
            eval_loss = MnistModel.evaluate_model(model, test_loader, criterion, device)
            print("Epoch: {}/{} | Eval Loss: {:.4f}".format(epoch+1, epochs, eval_loss))

        return model

    @staticmethod
    def evaluate_model(model, test_loader, criterion, device):
        model.eval()
        model.to(device)
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(test_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistModel().to(device)

    batch_size = 512
    epochs = 500
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = MnistModel.get_dataloaders(batch_size, num_workers=2)

    model = MnistModel.train_model(model, train_loader, test_loader, criterion, epochs, device, N=10)
    MnistModel.save_model(model, 'Mnist_nobackprop.pth')

if __name__ == "__main__":
    main()
