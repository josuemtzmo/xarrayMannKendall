from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='xarrayMannKendall',
    version='1.4.5',
    description='Mann-Kendall statistical test to assess if a monotonic upward or downward trend exists over time.',
    url='https://github.com/josuemtzmo/xarrayMannKendall',
    author='josuemtzmo',
    author_email='josue.martinezmoreno@anu.edu.au',
    license='MIT License',
    packages=find_packages(),
    install_requires=[],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
