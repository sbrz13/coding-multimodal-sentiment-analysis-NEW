from setuptools import setup, find_packages

setup(
    name="multimodal-sentiment-analysis",
    version="0.1.0",
    description="多模态情感分析框架，利用预训练模型进行特征融合",
    author="",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.5.1",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.41.2",
        "nltk==3.8.1",
        "Pillow==10.3.0",
        "opencv-python==4.10.0.84",
        "datasets==2.19.2",
        "matplotlib==3.9.1",
        "seaborn==0.13.2",
        "argparse==1.4.0",
        "tqdm==4.66.4"
    ],
    entry_points={
        "console_scripts": [
            "run_training=scripts.run_training:main",
            "run_evaluation=scripts.run_evaluation:main",
            "preprocess_data=scripts.preprocess_data:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8"
)
