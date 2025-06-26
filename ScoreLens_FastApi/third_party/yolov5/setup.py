from setuptools import setup, find_packages

setup(
    name='yolov5',
    version='7.0.0',
    description='YOLOv5 local package for object detection',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.2.2',
        'numpy>=1.18.5',
        'opencv-python>=4.1.2',
        'Pillow>=7.1.2',
        'PyYAML>=5.3.1',
        'scipy>=1.4.1',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tqdm>=4.41.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
)
