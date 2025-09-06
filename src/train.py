import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config, model_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )

        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # Criar diretório para salvar modelos
        os.makedirs("models", exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Treinando")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validando"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_acc = 100. * correct / total
        val_loss = total_loss / len(self.val_loader)

        return val_loss, val_acc, all_preds, all_targets

    def train(self):
        print(f"Iniciando treinamento do {self.model_name} por {self.config['epochs']} épocas...")

        for epoch in range(self.config['epochs']):
            print(f"\nÉpoca {epoch+1}/{self.config['epochs']}")

            # Treinamento
            train_loss, train_acc = self.train_epoch()

            # Validação
            val_loss, val_acc, preds, targets = self.validate()

            # Atualizar scheduler
            self.scheduler.step()

            # Salvar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Logging
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Salvar melhor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model_path = f"models/best_{self.model_name.lower()}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': val_acc,
                    'config': self.config
                }, model_path)
                print(f"Novo melhor modelo salvo! Acc: {val_acc:.2f}%")

            # Salvar checkpoint a cada 5 épocas
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"models/checkpoint_{self.model_name.lower()}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accs': self.train_accs,
                    'val_accs': self.val_accs,
                    'config': self.config
                }, checkpoint_path)

        # Salvar métricas finais
        self.save_training_metrics()
        self.plot_training_curves()

        return self.best_val_acc

    def save_training_metrics(self):
        """Salva as métricas de treinamento"""
        metrics = {
            'model_name': self.model_name,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        metrics_path = f"models/{self.model_name.lower()}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Métricas salvas em {metrics_path}")

    def plot_training_curves(self):
        """Plota as curvas de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title(f'Training and Validation Loss - {self.model_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.train_accs, label='Train Acc', color='green')
        ax2.plot(self.val_accs, label='Val Acc', color='orange')
        ax2.set_title(f'Training and Validation Accuracy - {self.model_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'models/{self.model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_model(model_name, model_path, train_loader, val_loader, device, config):
    """Função para treinar um modelo específico"""
    from .models import get_model

    print(f"\n{'='*50}")
    print(f"TREINANDO MODELO: {model_name}")
    print(f"{'='*50}")

    # Carregar modelo
    model = get_model(model_path, num_classes=len(train_loader.dataset.classes))

    # Criar trainer
    trainer = Trainer(model, train_loader, val_loader, device, config, model_name)

    # Treinar modelo
    best_acc = trainer.train()

    print(f"\nTreinamento do {model_name} concluído!")
    print(f"Melhor acurácia de validação: {best_acc:.2f}%")

    return trainer

def main():
    # Configurações
    config = {
        'batch_size': 16,  # Reduzido para evitar problemas de memória
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 2,  # Reduzido para demonstração
        'img_size': 224,
        'num_workers': 2  # Reduzido para Windows
    }

    # Carregar dados
    from .data_preparation import get_data_loaders

    data_dir = "data"
    train_loader, val_loader, classes = get_data_loaders(
        data_dir,
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers']
    )

    print(f"Dataset carregado:")
    print(f"  Classes: {len(classes)}")
    print(f"  Amostras de treinamento: {len(train_loader.dataset)}")
    print(f"  Amostras de validação: {len(val_loader.dataset)}")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Modelos para treinar (escolha 2 para comparar)
    from .models import RECOMMENDED_MODELS

    models_to_train = {
        "DeiT": RECOMMENDED_MODELS["DeiT"],
        "Swin": RECOMMENDED_MODELS["Swin"]
    }

    # Treinar modelos
    trainers = {}
    for model_name, model_path in models_to_train.items():
        try:
            trainer = train_model(model_name, model_path, train_loader, val_loader, device, config)
            trainers[model_name] = trainer
        except Exception as e:
            print(f"Erro ao treinar {model_name}: {e}")
            continue

    print(f"\n{'='*50}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*50}")

    # Resumo dos resultados
    for model_name, trainer in trainers.items():
        print(f"{model_name}: {trainer.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
