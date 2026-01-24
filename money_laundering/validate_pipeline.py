"""
Script de Validação do Pipeline de Produção

Este script verifica se todos os componentes do pipeline refatorado
estão funcionando corretamente.

Autor: TCC - Anti-Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from source.config import PROJ_ROOT, EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR


def validate_project_structure():
    """Valida se a estrutura de diretórios está correta."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 1: Estrutura do Projeto")
    logger.info("="*80)
    
    required_dirs = [
        EXTERNAL_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        PROJ_ROOT / 'source',
        PROJ_ROOT / 'source' / 'modeling',
        PROJ_ROOT / 'notebooks'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"{status} {dir_path.relative_to(PROJ_ROOT)}")
        if not exists:
            all_exist = False
    
    if all_exist:
        logger.success("✅ Estrutura de diretórios: OK\n")
        return True
    else:
        logger.error("❌ Alguns diretórios estão faltando!\n")
        return False


def validate_external_data():
    """Valida se os dados externos existem."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 2: Dados Externos")
    logger.info("="*80)
    
    # Verificar se existe pelo menos um dataset
    datasets = ['LI-Small', 'LI-Medium', 'LI-Large', 'HI-Small', 'HI-Medium', 'HI-Large']
    
    found_datasets = []
    for dataset in datasets:
        accounts_path = EXTERNAL_DATA_DIR / f'{dataset}_accounts.csv'
        trans_path = EXTERNAL_DATA_DIR / f'{dataset}_Trans.csv'
        
        if accounts_path.exists() and trans_path.exists():
            found_datasets.append(dataset)
            logger.info(f"✅ {dataset}: accounts.csv e Trans.csv encontrados")
    
    if found_datasets:
        logger.success(f"✅ {len(found_datasets)} dataset(s) disponível(is): {found_datasets}\n")
        return True
    else:
        logger.warning("⚠️  Nenhum dataset encontrado em data/external/")
        logger.info("   Baixe os dados e coloque-os em data/external/\n")
        return False


def validate_source_code():
    """Valida se os arquivos de código fonte existem."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 3: Código Fonte")
    logger.info("="*80)
    
    required_files = [
        PROJ_ROOT / 'source' / '__init__.py',
        PROJ_ROOT / 'source' / 'config.py',
        PROJ_ROOT / 'source' / 'dataset.py',
        PROJ_ROOT / 'source' / 'preprocessing.py',
        PROJ_ROOT / 'source' / 'modeling' / '__init__.py',
        PROJ_ROOT / 'source' / 'modeling' / 'train_pipeline.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"{status} {file_path.relative_to(PROJ_ROOT)}")
        if not exists:
            all_exist = False
    
    if all_exist:
        logger.success("✅ Arquivos de código fonte: OK\n")
        return True
    else:
        logger.error("❌ Alguns arquivos estão faltando!\n")
        return False


def validate_imports():
    """Valida se as importações principais funcionam."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 4: Importações")
    logger.info("="*80)
    
    imports_ok = True
    
    # Testar importação do preprocessador
    try:
        from source.preprocessing import AMLPreprocessor
        logger.info("✅ source.preprocessing.AMLPreprocessor")
    except Exception as e:
        logger.error(f"❌ Erro ao importar AMLPreprocessor: {e}")
        imports_ok = False
    
    # Testar importação do trainer
    try:
        from source.modeling.train_pipeline import AMLModelTrainer
        logger.info("✅ source.modeling.train_pipeline.AMLModelTrainer")
    except Exception as e:
        logger.error(f"❌ Erro ao importar AMLModelTrainer: {e}")
        imports_ok = False
    
    # Testar bibliotecas externas
    required_libs = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('imblearn', 'imblearn'),
        ('category_encoders', 'category_encoders'),
        ('xgboost', 'xgb'),
        ('lightgbm', 'lgb')
    ]
    
    for lib_name, import_name in required_libs:
        try:
            __import__(lib_name)
            logger.info(f"✅ {lib_name}")
        except ImportError:
            logger.warning(f"⚠️  {lib_name} não instalado")
            imports_ok = False
    
    if imports_ok:
        logger.success("✅ Todas as importações: OK\n")
        return True
    else:
        logger.warning("⚠️  Algumas importações falharam. Execute: pip install -r requirements.txt\n")
        return False


def validate_processed_data():
    """Valida se os dados processados existem."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 5: Dados Processados")
    logger.info("="*80)
    
    treino_path = PROCESSED_DATA_DIR / 'df_treino.csv'
    oot_path = PROCESSED_DATA_DIR / 'df_oot.csv'
    
    if treino_path.exists():
        logger.info(f"✅ {treino_path.name}")
    else:
        logger.warning(f"⚠️  {treino_path.name} não encontrado")
    
    if oot_path.exists():
        logger.info(f"✅ {oot_path.name}")
    else:
        logger.warning(f"⚠️  {oot_path.name} não encontrado")
    
    if treino_path.exists() and oot_path.exists():
        logger.success("✅ Dados processados: OK\n")
        return True
    else:
        logger.warning("⚠️  Execute: python source/dataset.py\n")
        return False


def validate_models():
    """Valida se os modelos treinados existem."""
    logger.info("="*80)
    logger.info("VALIDAÇÃO 6: Modelos Treinados")
    logger.info("="*80)
    
    preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
    
    if preprocessor_path.exists():
        logger.info(f"✅ {preprocessor_path.name}")
        has_models = True
    else:
        logger.warning(f"⚠️  {preprocessor_path.name} não encontrado")
        has_models = False
    
    # Verificar modelos
    model_files = list(MODELS_DIR.glob('*.pkl'))
    if model_files:
        for model_file in model_files:
            logger.info(f"✅ {model_file.name}")
    
    if has_models:
        logger.success(f"✅ {len(model_files)} modelo(s) encontrado(s)\n")
        return True
    else:
        logger.warning("⚠️  Execute: python source/modeling/train_pipeline.py\n")
        return False


def main():
    """Executa todas as validações."""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "VALIDAÇÃO DO PIPELINE DE PRODUÇÃO" + " "*25 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info(f"\nDiretório do Projeto: {PROJ_ROOT}\n")
    
    results = {
        'Estrutura': validate_project_structure(),
        'Dados Externos': validate_external_data(),
        'Código Fonte': validate_source_code(),
        'Importações': validate_imports(),
        'Dados Processados': validate_processed_data(),
        'Modelos': validate_models()
    }
    
    # Resumo final
    logger.info("="*80)
    logger.info("RESUMO DAS VALIDAÇÕES")
    logger.info("="*80)
    
    for validation, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        logger.info(f"{status:12s} | {validation}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info("\n" + "="*80)
    if total_passed == total_tests:
        logger.success(f"🎉 SUCESSO! {total_passed}/{total_tests} validações passaram!")
        logger.success("O pipeline está pronto para uso!")
    elif total_passed >= 4:
        logger.warning(f"⚠️  PARCIAL: {total_passed}/{total_tests} validações passaram")
        logger.info("Execute os comandos sugeridos acima para completar a configuração")
    else:
        logger.error(f"❌ ERRO: Apenas {total_passed}/{total_tests} validações passaram")
        logger.info("Revise os erros acima e corrija antes de prosseguir")
    
    logger.info("="*80)
    logger.info("\n")
    
    # Próximos passos
    if not results['Dados Processados']:
        logger.info("📌 PRÓXIMO PASSO: python source/dataset.py")
    elif not results['Modelos']:
        logger.info("📌 PRÓXIMO PASSO: python source/modeling/train_pipeline.py")
    else:
        logger.info("📌 PRÓXIMO PASSO: Analise os resultados em data/processed/model_results_oot.csv")
    
    logger.info("\n")


if __name__ == '__main__':
    main()
