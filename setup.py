from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="niftidicomconverter",
    version="0.1.0",
    author="Björn Ahlgren",
    author_email="bjorn.victor.ahlgren@gmail.com",
    description="A package for easy conversion between nifti and dicom file formats for medical imaging tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/my_deep_learning_package",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
    install_requires=[
        "nibabel>=4.0.2",
        "pydicom>=2.3.1"
    ],
    python_requires=">=3.7",
)
