from setuptools import setup, find_packages

setup(
    name='pinn_turbulent_channel_flow',
    version='1.0.0',
    description='PINN for RANS Channel Flow with k-epsilon turbulence model',
    author='PINN Research Team',
    author_email='contact@example.com',
    url='https://github.com/example/pinn_turbulent_channel_flow',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.12.0',
        'deepxde>=1.9.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0'
    ],
    extras_require={
        'dev': [
            'scikit-learn>=1.0.0',
            'seaborn>=0.11.0',
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'jupyterlab>=3.0.0'
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
