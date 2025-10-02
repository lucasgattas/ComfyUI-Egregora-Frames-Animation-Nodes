"""
Installation script for Egregora Frames Animation Nodes
Handles dependencies and optional AI model downloads
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if requirements_path.exists():
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            str(requirements_path)
        ])
        print("✓ Dependencies installed successfully")
    else:
        print("Warning: requirements.txt not found")

def check_optional_dependencies():
    """Check for optional AI interpolation models"""
    optional_packages = {
        "rife": "RIFE (AI Frame Interpolation)",
        "film": "FILM (Google Frame Interpolation)"
    }
    
    print("\nChecking optional AI models...")
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {description} - Available")
        except ImportError:
            print(f"✗ {description} - Not installed (optional)")
    
    print("\nNote: AI interpolation models are optional.")
    print("To install RIFE/FILM, see the documentation at:")
    print("https://github.com/yourusername/egregora-animation-nodes")

def download_models():
    """Download AI models if needed (placeholder for future implementation)"""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\nModels directory: {models_dir}")
    print("AI models will be downloaded here when needed.")
    
    # TODO: Implement automatic model downloading
    # - RIFE model weights
    # - FILM model weights
    # Check if models exist, if not, download from official sources

def main():
    """Main installation process"""
    print("=" * 60)
    print("Egregora Frames Animation Nodes - Installation")
    print("=" * 60)
    
    try:
        install_requirements()
        check_optional_dependencies()
        download_models()
        
        print("\n" + "=" * 60)
        print("Installation complete!")
        print("Restart ComfyUI to use the nodes.")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Installation failed: {e}")
        print("Please install dependencies manually:")
        print(f"  pip install -r {Path(__file__).parent}/requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()