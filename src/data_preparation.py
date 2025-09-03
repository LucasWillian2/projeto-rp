import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((str(class_dir / img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size=224):
    """Retorna as transformações para treinamento e validação"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """Retorna os data loaders para treinamento e validação"""
    train_transform, val_transform = get_transforms(img_size)

    train_dataset = PlantDiseaseDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = PlantDiseaseDataset(os.path.join(data_dir, 'valid'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes

if __name__ == "__main__":
    # Testar carregamento dos dados
    data_dir = "../data"
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size=4)

    print(f"Número de classes: {len(classes)}")
    print(f"Classes: {classes[:10]}...")  # Mostrar primeiras 10 classes

    # Verificar uma amostra
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Shape das imagens: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Classes correspondentes: {[classes[label] for label in labels]}")
        break
