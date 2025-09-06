import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import time
import os
import json
from datetime import datetime

class ModelComparator:
    def __init__(self, models, dataloader, device, class_names):
        self.models = models
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        self.results = {}

        # Criar diretório para resultados
        os.makedirs("model_comparison", exist_ok=True)

    def evaluate_model(self, model_name, model):
        """Avalia um modelo específico"""
        model.eval()
        all_preds = []
        all_targets = []
        inference_times = []
        all_probabilities = []

        print(f"Avaliando {model_name}...")

        with torch.no_grad():
            for images, targets in tqdm(self.dataloader, desc=f"Avaliando {model_name}"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                start_time = time.time()
                outputs = model(images)
                inference_time = time.time() - start_time

                # Obter predições e probabilidades
                probabilities = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                inference_times.extend([inference_time] * len(images))

        # Calcular métricas
        accuracy = accuracy_score(all_targets, all_preds)
        avg_inference_time = np.mean(inference_times)

        # Matriz de confusão
        cm = confusion_matrix(all_targets, all_preds)

        # Relatório de classificação
        report = classification_report(all_targets, all_preds, output_dict=True)

        # Calcular métricas por classe
        per_class_metrics = self.calculate_per_class_metrics(all_targets, all_preds, all_probabilities)

        return {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

    def calculate_per_class_metrics(self, targets, preds, probabilities):
        """Calcula métricas específicas por classe"""
        per_class = {}

        for i, class_name in enumerate(self.class_names):
            # Máscara para a classe atual
            mask = np.array(targets) == i

            if np.sum(mask) > 0:
                # Acurácia por classe
                class_acc = accuracy_score(
                    np.array(targets)[mask],
                    np.array(preds)[mask]
                )

                # Probabilidade média para predições corretas
                correct_probs = np.array(probabilities)[mask]
                correct_preds = np.array(preds)[mask] == i

                if np.sum(correct_preds) > 0:
                    avg_confidence = np.mean(correct_probs[correct_preds, i])
                else:
                    avg_confidence = 0.0

                per_class[class_name] = {
                    'accuracy': float(class_acc),
                    'avg_confidence': float(avg_confidence),
                    'samples': int(np.sum(mask))
                }

        return per_class

    def compare_models(self):
        """Compara todos os modelos"""
        print("Iniciando comparação de modelos...")

        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"AVALIANDO: {model_name}")
            print(f"{'='*50}")

            try:
                self.results[model_name] = self.evaluate_model(model_name, model)
                print(f"{model_name} avaliado com sucesso!")
            except Exception as e:
                print(f"Erro ao avaliar {model_name}: {e}")
                continue

        self.generate_comparison_report()

    def generate_comparison_report(self):
        """Gera relatório completo de comparação"""
        if not self.results:
            print("Nenhum resultado para gerar relatório.")
            return

        # Tabela de métricas principais
        metrics_df = pd.DataFrame({
            'Modelo': list(self.results.keys()),
            'Acurácia': [f"{self.results[name]['accuracy']:.4f}" for name in self.results.keys()],
            'Tempo de Inferência (s)': [f"{self.results[name]['avg_inference_time']:.4f}" for name in self.results.keys()]
        })

        print("\n" + "="*60)
        print("RELATÓRIO DE COMPARAÇÃO DE MODELOS")
        print("="*60)
        print(metrics_df.to_string(index=False))

        # Gráficos de comparação
        self.plot_comparison()

        # Análise detalhada por classe
        self.analyze_per_class_performance()

        # Salvar resultados
        self.save_results()

        print(f"\nRelatório completo salvo em 'model_comparison/'")

    def plot_comparison(self):
        """Plota gráficos de comparação"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        models = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in models]
        inference_times = [self.results[name]['avg_inference_time'] for name in models]

        # 1. Acurácia
        bars1 = axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 0].set_title('Comparação de Acurácia', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Adicionar valores nas barras
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Tempo de inferência
        bars2 = axes[0, 1].bar(models, inference_times, color=['lightcoral', 'skyblue', 'gold', 'lightgreen'])
        axes[0, 1].set_title('Tempo de Inferência', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Tempo (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Matriz de confusão do melhor modelo
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_model]['confusion_matrix']

        # Normalizar matriz de confusão
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Matriz de Confusão Normalizada\n{best_model} (Melhor)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predito')
        axes[1, 0].set_ylabel('Real')

        # 4. Comparação de acurácia por classe
        self.plot_per_class_accuracy(axes[1, 1])

        plt.tight_layout()
        plt.savefig('model_comparison/model_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_per_class_accuracy(self, ax):
        """Plota acurácia por classe para todos os modelos"""
        # Selecionar algumas classes para visualização (primeiras 10)
        classes_to_plot = self.class_names[:10]

        x = np.arange(len(classes_to_plot))
        width = 0.8 / len(self.results)

        for i, (model_name, result) in enumerate(self.results.items()):
            accuracies = [result['per_class_metrics'].get(cls, {}).get('accuracy', 0) for cls in classes_to_plot]
            ax.bar(x + i * width, accuracies, width, label=model_name, alpha=0.8)

        ax.set_title('Acurácia por Classe (Top 10)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Acurácia')
        ax.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax.set_xticklabels(classes_to_plot, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def analyze_per_class_performance(self):
        """Analisa performance por classe"""
        print(f"\n{'='*60}")
        print("ANÁLISE DETALHADA POR CLASSE")
        print(f"{'='*60}")

        # Encontrar melhor modelo
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        print(f"Modelo com melhor performance: {best_model}")

        # Análise das classes com melhor e pior performance
        per_class_metrics = self.results[best_model]['per_class_metrics']

        # Ordenar por acurácia
        sorted_classes = sorted(per_class_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"\nTop 5 classes com melhor performance:")
        for i, (class_name, metrics) in enumerate(sorted_classes[:5]):
            print(f"  {i+1}. {class_name}: {metrics['accuracy']:.4f} ({metrics['samples']} amostras)")

        print(f"\nTop 5 classes com pior performance:")
        for i, (class_name, metrics) in enumerate(sorted_classes[-5:]):
            print(f"  {i+1}. {class_name}: {metrics['accuracy']:.4f} ({metrics['samples']} amostras)")

    def save_results(self):
        """Salva todos os resultados"""
        # Converter numpy arrays para listas para serialização
        results_to_save = {}
        for model_name, result in self.results.items():
            # Converter per_class_metrics para tipos Python nativos
            per_class_metrics_clean = {}
            for class_name, metrics in result['per_class_metrics'].items():
                per_class_metrics_clean[class_name] = {
                    'accuracy': float(metrics['accuracy']),
                    'avg_confidence': float(metrics['avg_confidence']),
                    'samples': int(metrics['samples'])
                }

            results_to_save[model_name] = {
                'accuracy': float(result['accuracy']),
                'avg_inference_time': float(result['avg_inference_time']),
                'confusion_matrix': result['confusion_matrix'].tolist(),
                'classification_report': result['classification_report'],
                'per_class_metrics': per_class_metrics_clean
            }

        # Salvar resultados principais
        with open('model_comparison/model_comparison_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)

        # Salvar resumo em CSV
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Modelo': model_name,
                'Acurácia': result['accuracy'],
                'Tempo_Inferencia_s': result['avg_inference_time'],
                'Timestamp': datetime.now().isoformat()
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('model_comparison/model_comparison_summary.csv', index=False)

        print(f"Resultados salvos em 'model_comparison/'")

def load_trained_models(model_names, models_dir="models"):
    """Carrega modelos treinados"""
    from .models import get_model, RECOMMENDED_MODELS

    loaded_models = {}

    for model_name in model_names:
        try:
            # Caminho para os pesos treinados
            weight_path = os.path.join(models_dir, f"best_{model_name.lower()}.pth")

            if os.path.exists(weight_path):
                # Carregar modelo
                model_path = RECOMMENDED_MODELS[model_name]
                model = get_model(model_path, num_classes=38)  # 38 classes do dataset

                # Carregar pesos treinados
                checkpoint = torch.load(weight_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                loaded_models[model_name] = model
                print(f"Modelo {model_name} carregado com sucesso de {weight_path}")
            else:
                print(f"Pesos não encontrados para {model_name} em {weight_path}")

        except Exception as e:
            print(f"Erro ao carregar {model_name}: {e}")

    return loaded_models

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carregar dados
    from .data_preparation import get_data_loaders

    data_dir = "data"
    _, val_loader, classes = get_data_loaders(
        data_dir,
        batch_size=16,
        num_workers=2
    )

    print(f"Dataset carregado:")
    print(f"  Classes: {len(classes)}")
    print(f"  Amostras de validação: {len(val_loader.dataset)}")

    # Modelos para comparar (escolha 2)
    model_names_to_compare = ["ViT", "Swin"]

    # Carregar modelos treinados
    print(f"\nCarregando modelos treinados...")
    models = load_trained_models(model_names_to_compare)

    if not models:
        print("Nenhum modelo foi carregado. Treine os modelos primeiro usando 'python src/train.py'")
        return

    # Comparar modelos
    print(f"\nIniciando comparação de {len(models)} modelos...")
    comparator = ModelComparator(models, val_loader, device, classes)
    comparator.compare_models()

if __name__ == "__main__":
    main()
