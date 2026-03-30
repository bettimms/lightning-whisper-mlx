from setuptools import setup, find_packages

setup(
    name='lightning-whisper-mlx',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'lightning_whisper_mlx': ['assets/*']
    },
    install_requires=[
        'huggingface_hub',
        "mlx",
        "numba",
        "numpy",
        "tqdm",
        "more-itertools",
        "tiktoken==0.3.3",
        "scipy",
    ],
    extras_require={
        "fast-audio": ["soundfile", "librosa"],
    },
    python_requires=">=3.10",
)