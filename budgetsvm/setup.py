from setuptools import setup, find_packages

setup(
    name="budgetsvm",
    version='0.0.1',
    url='',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy==1.23.5', 'gurobipy==10.0.1', 'scikit_learn==1.2.2'],
)