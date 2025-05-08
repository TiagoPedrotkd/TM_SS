"""
Test script to verify the environment setup and required packages.
"""

def test_imports():
    """Test importing all required packages."""
    # First test PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {str(e)}")
        return

    # Then test transformers with PyTorch
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import transformers: {str(e)}")
        return

    # Test remaining packages
    packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'sklearn': 'sklearn',
        'nltk': 'nltk',
        'plotly': 'plotly',
        'wordcloud': 'WordCloud',
        'tqdm': 'tqdm',
        'gensim': 'gensim'
    }
    
    failed_imports = []
    for package, alias in packages.items():
        try:
            exec(f"import {package} as {alias}")
            print(f"✓ Successfully imported {package}")
        except ImportError as e:
            failed_imports.append(f"✗ Failed to import {package}: {str(e)}")
    
    if failed_imports:
        print("\nSome imports failed:")
        for error in failed_imports:
            print(error)
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
    else:
        print("\nAll required packages are installed correctly!")

if __name__ == "__main__":
    print("Testing environment setup...")
    test_imports() 