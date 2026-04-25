from setuptools import setup, find_packages

setup(
    name="multimodal-sentiment-analysis",
    version="0.2.0",
    description="Multi-level feature fusion framework for multimodal sentiment analysis with transfer learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "nltk>=3.8.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "datasets>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "run_training=scripts.run_training:main",
            "run_evaluation=scripts.run_evaluation:main",
            "preprocess_data=scripts.preprocess_data:main",
        ]
    },
    python_requires=">=3.8",
)
