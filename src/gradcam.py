import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks para capturar gradientes e ativações
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target = self.target_layer
        self.hooks.append(target.register_forward_hook(forward_hook))
        self.hooks.append(target.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, input_image, target_class):
        """Gera o mapa de ativação GradCAM"""
        # Forward pass
        model_output = self.model(input_image)

        if target_class is None:
            target_class = model_output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        model_output[0, target_class].backward()

        # Calcular GradCAM
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            raise ValueError("Gradientes ou ativações não foram capturados. Verifique se o target_layer está correto.")

        weights = torch.mean(gradients, dim=[2, 3])
        cam = torch.sum(weights[:, :, None, None] * activations, dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0)

        # Normalizar
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def visualize(self, input_image, target_class=None, save_path=None, class_names=None):
        """Visualiza o GradCAM sobreposto na imagem"""
        # Gerar CAM
        cam = self.generate_cam(input_image, target_class)

        # Converter para numpy
        cam_np = cam.detach().cpu().numpy()

        # Redimensionar para o tamanho da imagem original
        cam_resized = cv2.resize(cam_np, (input_image.shape[3], input_image.shape[2]))

        # Normalizar para 0-255
        cam_normalized = (cam_resized * 255).astype(np.uint8)

        # Aplicar colormap
        cam_colored = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)

        # Converter imagem de entrada para visualização
        img_np = input_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)

        # Sobrepor CAM na imagem
        cam_resized_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        cam_resized_rgb = cv2.resize(cam_resized_rgb, (img_np.shape[1], img_np.shape[0]))

        # Misturar imagem original com CAM
        alpha = 0.6
        overlay = cv2.addWeighted(img_np, 1-alpha, cam_resized_rgb, alpha, 0)

        # Visualizar
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Imagem original
        axes[0].imshow(img_np)
        if class_names and target_class is not None:
            class_name = class_names[target_class] if target_class < len(class_names) else f"Class {target_class}"
            axes[0].set_title(f'Imagem Original\nClasse: {class_name}')
        else:
            axes[0].set_title('Imagem Original')
        axes[0].axis('off')

        # GradCAM
        axes[1].imshow(cam_normalized, cmap='jet')
        axes[1].set_title('GradCAM\n(Áreas de Atenção)')
        axes[1].axis('off')

        # Sobreposição
        axes[2].imshow(overlay)
        axes[2].set_title('Sobreposição\n(GradCAM + Imagem)')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return cam_np, overlay

def find_target_layer(model):
    """Encontra a camada alvo para GradCAM"""
    target_layer = None

    # Tentar encontrar camadas específicas
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['classifier', 'head', 'fc', 'linear']):
            target_layer = module
            print(f"Camada alvo encontrada: {name}")
            break

    # Se não encontrar, usar a última camada
    if target_layer is None:
        target_layer = list(model.modules())[-1]
        print(f"Usando última camada como alvo: {type(target_layer).__name__}")

    return target_layer

def apply_gradcam_to_dataset(model, dataloader, target_layer, num_samples=10, save_dir="gradcam_results", class_names=None):
    """Aplica GradCAM em várias amostras do dataset"""
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model, target_layer)

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_samples:
            break

        for j in range(images.shape[0]):
            img = images[j:j+1]
            label = labels[j]

            # Gerar visualização
            try:
                cam, overlay = gradcam.visualize(
                    img,
                    target_class=label,
                    save_path=f"{save_dir}/sample_{i}_{j}_class_{label}.png",
                    class_names=class_names
                )
                print(f"GradCAM gerado para amostra {i}_{j}, classe {label}")
            except Exception as e:
                print(f"Erro ao gerar GradCAM para amostra {i}_{j}: {e}")

    gradcam.remove_hooks()
    print(f"GradCAM aplicado em {num_samples} amostras. Resultados salvos em {save_dir}")

def compare_gradcam_models(models, dataloader, num_samples=5, save_dir="gradcam_comparison"):
    """Compara GradCAM entre diferentes modelos"""
    os.makedirs(save_dir, exist_ok=True)

    # Pegar algumas amostras
    sample_images = []
    sample_labels = []

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_samples:
            break
        sample_images.append(images[0:1])  # Primeira imagem do batch
        sample_labels.append(labels[0])

    # Aplicar GradCAM para cada modelo
    for model_name, model in models.items():
        print(f"\nAplicando GradCAM para {model_name}...")

        # Encontrar camada alvo
        target_layer = find_target_layer(model)

        # Criar GradCAM
        gradcam = GradCAM(model, target_layer)

        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            try:
                # Gerar visualização
                cam, overlay = gradcam.visualize(
                    img,
                    target_class=label,
                    save_path=f"{save_dir}/{model_name}_sample_{i}_class_{label}.png"
                )
            except Exception as e:
                print(f"Erro ao gerar GradCAM para {model_name}, amostra {i}: {e}")

        gradcam.remove_hooks()

    print(f"\nComparação de GradCAM concluída! Resultados salvos em {save_dir}")

# Exemplo de uso
if __name__ == "__main__":
    from models import get_model, RECOMMENDED_MODELS
    from data_preparation import get_data_loaders

    # Carregar dados
    data_dir = "data"
    _, val_loader, classes = get_data_loaders(data_dir, batch_size=4, num_workers=2)

    # Carregar modelo treinado (se existir)
    model_name = "ViT"
    model_path = RECOMMENDED_MODELS[model_name]

    try:
        model = get_model(model_path, num_classes=len(classes))

        # Carregar pesos treinados se existirem
        weight_path = f"models/best_{model_name.lower()}.pth"
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Pesos carregados de {weight_path}")

        model.eval()

        # Encontrar camada alvo
        target_layer = find_target_layer(model)

        # Testar GradCAM
        gradcam = GradCAM(model, target_layer)

        # Pegar uma amostra do dataset
        for images, labels in val_loader:
            img = images[0:1]
            label = labels[0]

            print(f"Testando GradCAM para classe: {classes[label]}")

            # Gerar visualização
            cam, overlay = gradcam.visualize(
                img,
                target_class=label,
                save_path=f"gradcam_test_{model_name}.png",
                class_names=classes
            )
            break

        gradcam.remove_hooks()

    except Exception as e:
        print(f"Erro ao testar GradCAM: {e}")
        print("Certifique-se de que o modelo foi treinado primeiro.")
