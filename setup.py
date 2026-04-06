from setuptools import find_namespace_packages, setup

setup(
    name="plant-disease-mlops",
    version="0.1.0",
    description="Plant disease detection MLOps pipeline",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "boto3>=1.28.0",
        "datasets>=2.16.0",
        "great-expectations>=1.0.0",
        "huggingface_hub>=0.20.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.65.0",
    ],
)
