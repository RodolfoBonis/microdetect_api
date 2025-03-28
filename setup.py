from setuptools import setup, find_packages

setup(
    name="microdetect",
    version="1.4.46",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.103.1",
        "uvicorn>=0.22.0",
        "SQLAlchemy>=2.0.20",
        "python-multipart>=0.0.5",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.1",
        "scipy>=1.10.0",
        "alembic>=1.11.0",
        "python-dotenv>=1.0.0",
        "psutil>=5.9.5",
        "pytz>=2023.3",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.32.3",
        "GPUtil>=1.4.0",
    ],
    entry_points={
        'console_scripts': [
            'microdetect=microdetect.cli:main',
        ],
    },
    python_requires=">=3.8",
    author="Sua Empresa",
    author_email="contato@empresa.com",
    description="Backend Python para detecção de microorganismos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/suaempresa/microdetect",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)