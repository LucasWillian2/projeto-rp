import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

class HuggingFaceModel(nn.Module):
    """Modelo base para modelos do Hugging Face Transformers"""
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super().__init__()
        self.processor, self.model = _load_model_components(model_name, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Congela todos os parâmetros exceto a camada de classificação"""
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongelar apenas a camada de classificação
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x).logits

VisionTransformer = HuggingFaceModel

def get_model(model_name, num_classes, model_type="huggingface"):
    """Retorna o modelo configurado"""
    if model_type == "huggingface":
        return HuggingFaceModel(model_name, num_classes)
    elif model_type == "vision_transformer":
        return VisionTransformer(model_name, num_classes)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")


RECOMMENDED_MODELS = {
    "ViT": "google/vit-base-patch16-224",
    "Swin": "microsoft/swin-tiny-patch4-window7-224",
    "ConvNeXt": "facebook/convnext-tiny-224",
    "DeiT": "facebook/deit-base-distilled-patch16-224"
}

def _load_model_components(model_name, num_classes=None):
    """Função auxiliar para carregar componentes do modelo"""
    processor = AutoImageProcessor.from_pretrained(model_name)

    if num_classes:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(model_name)

    return processor, model

def get_model_info(model_name):
    """Retorna informações sobre o modelo"""
    try:
        processor, model = _load_model_components(model_name)

        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'processor': processor,
            'model': model
        }
    except Exception as e:
        print(f"Erro ao carregar {model_name}: {e}")
        return None

if __name__ == "__main__":
    # Testar carregamento de um modelo
    print("Testando carregamento de modelo...")

    model_name = RECOMMENDED_MODELS["ViT"]
    model_info = get_model_info(model_name)

    if model_info:
        print(f"Modelo: {model_info['name']}")
        print(f"Total de parâmetros: {model_info['total_params']:,}")
        print(f"Parâmetros treináveis: {model_info['trainable_params']:,}")

        # Testar com modelo para 38 classes
        model = get_model(model_name, num_classes=38)
        print(f"Modelo configurado para {38} classes")

        # Testar forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
    else:
        print("Falha ao carregar modelo")
