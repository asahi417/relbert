from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

VERSION = '0.0.0'
setup(
    name='relbert',
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    license='MIT',
    description='TBA',
    url='https://github.com/asahi417/relbert',
    # download_url="https://github.com/asahi417/relbert/archive/v{}.tar.gz".format(VERSION),
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
        "gensim>=3.8.3",
        "pandas",
        "numpy",
        "transformers",
        "sentencepiece",
        "tensorboard",
        "truecase"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'relbert-train = relbert_cli.train:main',
            'relbert-eval = relbert_cli.evaluate:main',
            'relbert-prompt = relbert_cli.prompt:main'
        ]
    }
)

