import setuptools
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UGFraud", # Replace with your own username
    version="0.1.1.2",
    author="Yingtong Dou, Chen Wang, Sihong Xie, Guixiang Ma, and UIC BDSC Lab",
    author_email="bdscsafegraph@gmail.com",
    description="An Unsupervised Graph-based Toolbox for Fraud Detection",
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
    url="https://github.com/safe-graph/UGFraud",
    download_url='https://github.com/safe-graph/UGFraud/archive/master.zip',
    keywords=['fraud detection', 'anomaly detection', 'graph algorithm',
                'data mining', 'security'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "UGFraud": ["Yelp_Data/YelpChi/*.gz", "Yelp_Data/YelpChi/*.pkl"]},
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
