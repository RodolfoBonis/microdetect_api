from setuptools import setup, find_packages

setup(
    name="microdetect",
    version="1.4.41",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "sqlalchemy>=2.0.23",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.3",
        "python-multipart>=0.0.6",
        "numpy>=1.25.0",
        "alembic>=1.12.0",
        "psutil>=5.9.5",
        "pillow>=10.0.1",
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