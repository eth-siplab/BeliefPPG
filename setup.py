from setuptools import setup, find_packages

setup(
    name='beliefppg',
    version='0.1',
    packages=find_packages(),
    package_data={'beliefppg': ['inference/*.keras']},
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
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/eth-siplab/BeliefPPG',
    classifiers=[
        'Development Status :: 3 - Alpha',
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
