from setuptools import setup, find_packages

with open('README_PyPI.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='beliefppg',
    version='0.2.0',
    packages=find_packages(),
    package_data={'beliefppg': ['inference/inference_model.keras', 'inference/inference_model_notimebackbone.keras']},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow~=2.14.0',
        'tensorflow-probability~=0.22.1',
        'wandb'
    ],
    python_requires='>=3.9',
    author='Paul Streli',
    author_email='paul.streli@inf.ethz.ch',
    description='Taking multi-channel PPG and Accelerometer signals as input, BeliefPPG predicts the instantaneous heart rate and provides an uncertainty estimate for the prediction.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/eth-siplab/BeliefPPG',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='PPG, heart rate, signal processing, uncertainty estimation, biomedical signals',
)
