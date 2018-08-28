from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='nnCompression',
      version='0.1',
      description='Compress deep neural networks',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
      ],
      url='',
      author='',
      author_email='',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.9.1',
          'keras>=2'
      ])