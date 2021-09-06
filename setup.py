from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

VERSION = '0.0.1'
setup(
    name='relbert',
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    license='MIT',
    description='RelBERT: the state-of-the-art lexical relation embedding model.',
    url='https://github.com/asahi417/relbert',
    download_url="https://github.com/asahi417/relbert/archive/v{}.tar.gz".format(VERSION),
    keywords=['nlp'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch",
        "tqdm",
        "requests",
        "tfrecord",
        "tensorboard",
        "gdown",
        "pandas",
        "numpy",
        "transformers",
        "sentencepiece",
        "tensorboard",
        "truecase",
        "sklearn",
        "gensim==3.8.1"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'relbert-train = relbert_cl.train:main',
            'relbert-eval = relbert_cl.evaluate:main',
            'relbert-prompt-discrete = relbert_cl.prompt_discrete:main',
            'relbert-prompt-continuous = relbert_cl.prompt_continuous:main'
        ]
    }
)

