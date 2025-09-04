#!/usr/bin/env python3
"""
Auto-detect GPU and install appropriate PyTorch environment
"""

import subprocess
import sys


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        # Try nvidia-ml-py or nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_appropriate_environment():
    """Install the appropriate pixi environment"""
    if has_nvidia_gpu():
        print("🎯 NVIDIA GPU detected. Installing GPU environment with CUDA 12.6...")
        cmd = ["pixi", "install", "--environment", "default"]
    else:
        print("💻 No NVIDIA GPU detected. Installing CPU environment...")
        cmd = ["pixi", "install", "--environment", "cpu"]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Installation completed successfully!")

        # Verify installation
        verify_cmd = ["pixi", "run", "check-gpu"]
        subprocess.run(verify_cmd)

    except subprocess.CalledProcessError as e:
        if has_nvidia_gpu():
            print("❌ GPU installation failed. Falling back to CPU...")
            fallback_cmd = ["pixi", "install", "--environment", "cpu"]
            subprocess.run(fallback_cmd, check=True)
            print("✅ CPU fallback installation completed!")
        else:
            print(f"❌ Installation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    install_appropriate_environment()
