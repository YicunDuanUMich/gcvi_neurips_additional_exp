from setuptools import setup, find_packages

setup(
    name="gcvi",
    version="0.1.0",
    author="Declan Mcnamara",
    description="The implementaion of GCVI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/declanmcnamara/gcvi_neurips",
    packages=find_packages(),
    python_requires=">=3.7",
)
