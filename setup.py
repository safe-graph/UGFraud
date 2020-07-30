import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="UGFraud", # Replace with your own username
    version="0.0.1",
    author="Yingtong Dou, Chen Wang, Sihong Xie, Guixiang Ma, and UIC BDSC Lab",
    author_email="bdscsafegraph@gmail.com",
    description="An Unsupervised Graph-based Toolbox for Fraud Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safe-graph/UGFraud",
    download_url='https://github.com/safe-graph/UGFraud/archive/master.zip',
    keywords=['fraud detection', 'anomaly detection', 'graph algorithm',
                'data mining', 'security'],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
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