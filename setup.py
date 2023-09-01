from setuptools import find_packages, setup

setup(
    name="text_classifier",
    py_modules=["text_classifier"],
    version="1.0.0",
    description="Text classification python module.",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3.9"],
)
