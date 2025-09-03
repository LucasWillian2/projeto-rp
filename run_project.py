#!/usr/bin/env python3
"""
🌱 Script Principal - Detecção de Doenças em Plantas

Este script executa todo o pipeline do projeto:
1. Preparação dos dados
2. Treinamento dos modelos
3. Comparação de performance
4. Visualização com GradCAM

Uso:
    python run_project.py --mode all          # Executa tudo
    python run_project.py --mode train        # Apenas treinamento
    python run_project.py --mode compare      # Apenas comparação
    python run_project.py --mode gradcam      # Apenas GradCAM
"""

import argparse
import os
import sys
import time
from pathlib import Path

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    print("🔍 Verificando dependências...")

    required_packages = [
        'torch', 'torchvision', 'transformers', 'datasets',
        'PIL', 'numpy', 'matplotlib', 'seaborn', 'sklearn',
        'cv2', 'tqdm'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")

    if missing_packages:
        print(f"\n❌ Pacotes faltando: {', '.join(missing_packages)}")
        print("   Execute: pip install -r requirements.txt")
        return False

    print("✅ Todas as dependências estão instaladas!")
    return True

def check_dataset():
    """Verifica se o dataset está disponível"""
    print("\n📊 Verificando dataset...")

    data_dir = Path("data")
    required_folders = ["train", "valid", "test"]

    if not data_dir.exists():
        print("   ❌ Pasta 'data/' não encontrada")
        print("   📥 Baixe o dataset do Kaggle primeiro")
        return False

    missing_folders = []
    for folder in required_folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            missing_folders.append(folder)
            print(f"   ❌ Pasta '{folder}/' não encontrada")
        else:
            # Verificar se tem subpastas (classes)
            subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
            if len(subfolders) == 0:
                print(f"   ⚠️  Pasta '{folder}/' está vazia")
            else:
                print(f"   ✅ {folder}/ ({len(subfolders)} classes)")

    if missing_folders:
        print(f"\n❌ Pastas faltando: {', '.join(missing_folders)}")
        return False

    print("✅ Dataset verificado com sucesso!")
    return True

def run_data_preparation():
    """Executa a preparação dos dados"""
    print("\n📊 Executando preparação dos dados...")

    try:
        sys.path.append('src')
        from data_preparation import get_data_loaders

        # Testar carregamento
        train_loader, val_loader, classes = get_data_loaders(
            "data",
            batch_size=4,
            num_workers=2
        )

        print(f"   ✅ Dados carregados: {len(classes)} classes")
        print(f"   📚 Treinamento: {len(train_loader.dataset)} amostras")
        print(f"   ✅ Validação: {len(val_loader.dataset)} amostras")

        return True

    except Exception as e:
        print(f"   ❌ Erro na preparação dos dados: {e}")
        return False

def run_training():
    """Executa o treinamento dos modelos"""
    print("\n🚀 Executando treinamento dos modelos...")

    try:
        # Verificar se já existem modelos treinados
        models_dir = Path("models")
        if models_dir.exists():
            existing_models = list(models_dir.glob("best_*.pth"))
            if existing_models:
                print("   ⚠️  Modelos treinados já existem:")
                for model in existing_models:
                    print(f"      - {model.name}")

                response = input("   🔄 Deseja sobrescrever? (y/N): ").lower()
                if response != 'y':
                    print("   ⏭️  Pulando treinamento...")
                    return True

        # Executar treinamento
        from src.train import main as train_main
        train_main()

        return True

    except Exception as e:
        print(f"   ❌ Erro no treinamento: {e}")
        return False

def run_comparison():
    """Executa a comparação dos modelos"""
    print("\n📈 Executando comparação dos modelos...")

    try:
        # Verificar se existem modelos treinados
        models_dir = Path("models")
        if not models_dir.exists():
            print("   ❌ Pasta 'models/' não encontrada")
            print("   🚀 Execute o treinamento primeiro")
            return False

        existing_models = list(models_dir.glob("best_*.pth"))
        if not existing_models:
            print("   ❌ Nenhum modelo treinado encontrado")
            print("   🚀 Execute o treinamento primeiro")
            return False

        print(f"   ✅ {len(existing_models)} modelos encontrados")

        # Executar comparação
        from src.model_comparison import main as compare_main
        compare_main()

        return True

    except Exception as e:
        print(f"   ❌ Erro na comparação: {e}")
        return False

def run_gradcam():
    """Executa a visualização com GradCAM"""
    print("\n🔍 Executando visualização com GradCAM...")

    try:
        # Verificar se existem modelos treinados
        models_dir = Path("models")
        if not models_dir.exists():
            print("   ❌ Pasta 'models/' não encontrada")
            print("   🚀 Execute o treinamento primeiro")
            return False

        existing_models = list(models_dir.glob("best_*.pth"))
        if not existing_models:
            print("   ❌ Nenhum modelo treinado encontrado")
            print("   🚀 Execute o treinamento primeiro")
            return False

        print(f"   ✅ {len(existing_models)} modelos encontrados")

        # Executar GradCAM
        from src.gradcam import main as gradcam_main
        gradcam_main()

        return True

    except Exception as e:
        print(f"   ❌ Erro no GradCAM: {e}")
        return False

def run_notebook():
    """Abre o notebook Jupyter"""
    print("\n📓 Abrindo notebook Jupyter...")

    notebook_path = Path("notebooks/plant_disease_detection.ipynb")
    if not notebook_path.exists():
        print("   ❌ Notebook não encontrado")
        return False

    try:
        import subprocess
        subprocess.run(["jupyter", "notebook", str(notebook_path)])
        return True
    except Exception as e:
        print(f"   ❌ Erro ao abrir notebook: {e}")
        print("   💡 Execute manualmente: jupyter notebook notebooks/plant_disease_detection.ipynb")
        return False

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="🌱 Script Principal - Detecção de Doenças em Plantas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_project.py --mode all          # Executa todo o pipeline
  python run_project.py --mode train        # Apenas treinamento
  python run_project.py --mode compare      # Apenas comparação
  python run_project.py --mode gradcam      # Apenas GradCAM
  python run_project.py --mode notebook     # Abre o notebook
        """
    )

    parser.add_argument(
        '--mode',
        choices=['all', 'train', 'compare', 'gradcam', 'notebook'],
        default='all',
        help='Modo de execução (padrão: all)'
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Pular verificações de dependências e dataset'
    )

    args = parser.parse_args()

    print("🌱" + "="*60)
    print("   DETECÇÃO DE DOENÇAS EM PLANTAS - PIPELINE COMPLETO")
    print("="*60)

    start_time = time.time()

    # Verificações iniciais
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)

        if not check_dataset():
            print("\n❌ Dataset não encontrado ou incompleto")
            print("📥 Baixe o dataset do Kaggle primeiro:")
            print("   https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
            sys.exit(1)

    # Executar pipeline baseado no modo
    success = True

    if args.mode in ['all', 'train']:
        if not run_data_preparation():
            success = False
        elif not run_training():
            success = False

    if args.mode in ['all', 'compare'] and success:
        if not run_comparison():
            success = False

    if args.mode in ['all', 'gradcam'] and success:
        if not run_gradcam():
            success = False

    if args.mode == 'notebook':
        run_notebook()

    # Resumo final
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE EXECUTADO COM SUCESSO!")
        print(f"⏱️  Tempo total: {duration:.1f} segundos")

        if args.mode == 'all':
            print("\n📁 Arquivos gerados:")
            print("   📂 models/ - Modelos treinados")
            print("   📂 model_comparison/ - Resultados da comparação")
            print("   📂 gradcam_results/ - Visualizações do GradCAM")
            print("\n💡 Próximos passos:")
            print("   📓 Abra o notebook: python run_project.py --mode notebook")
            print("   🔍 Analise os resultados em model_comparison/")
            print("   📊 Visualize as curvas de treinamento em models/")
    else:
        print("❌ PIPELINE FALHOU!")
        print("🔍 Verifique os erros acima e tente novamente")

    print("="*60)

if __name__ == "__main__":
    main()
