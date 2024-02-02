from setuptools import setup, find_packages

setup(
    name='RLK_Unet',
    version='1.0',
    description='Tool for brain metastasis(BMs) segmentation',
    author='Seungyeon Son',
    author_email='airis@hanyang.ac.kr',
    license='Apache 2.0',
    url='https://github.com/nibabel/rlk_unet',
    install_requires=[
        'numpy>=1.14.5',
        'nibabel',
        'scipy',
        'timm',
        'torch>=1.9.0',
        ],
    scripts=['RLK_Unet/rlk_unet.py'],
    packages=find_packages(include=['RLK_Unet']),
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Linux',
    ],
)