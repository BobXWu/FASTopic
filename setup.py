from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

VERSION = '0.0.5'
DESCRIPTION = 'FASTopic'
LONG_DESCRIPTION = """
    FASTopic is a topic modeling package.
    It leverages pretrained transformers to produce document embeddings, and discovers latent topics through the optimal transport between docment, topic, and word embeddings.
"""


# Setting up
setup(
        name="fastopic",
        version=VERSION,
        author="Xiaobao Wu",
        author_email="xiaobao002@e.ntu.edu.sg",
        description=DESCRIPTION,
        long_description=readme,
        # long_description_content_type="text/x-rst",
        long_description_content_type="text/markdown",
        url='https://github.com/bobxwu/FASTopic',
        packages=find_packages(),
        license="Apache 2.0 License",
        install_requires=requirements,
        keywords=['topic model', 'neural topic model', 'transformers', 'optimal transport'],
        include_package_data=True,
        test_suite='tests',
        classifiers= [
            'Development Status :: 3 - Alpha',
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
