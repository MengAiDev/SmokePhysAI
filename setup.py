from setuptools import setup, find_packages

setup(
    name="smokephysai",
    version="1.0.0",
    description="Physics-aware AI vision model using incense smoke dynamics",
    author="SmokePhysAI Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "opencv-python-headless>=4.5.0",
        "Pillow>=8.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)