import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UGFraud", # Replace with your own username
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="An Unsupervised Graph-based Toolbox for Fraud Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safe-graph/UGFraud",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)