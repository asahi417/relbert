from setuptools import setup, find_packages

VERSION = '2.0.2'

with open('README.md', 'r') as f:
    readme = f.read()

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
    extras_require={
        "visualization": [
            'umap-learn',
            'hdbscan',
            'seaborn',
            'gensim==3.8.1'
        ]
    },
    install_requires=[
        "torch",
        "tqdm",
        "pandas",
        "numpy==1.23.5",  # https://stackoverflow.com/questions/74947992/how-to-remove-the-error-systemerror-initialization-of-internal-failed-without
        "transformers<=4.21.2",  # push-to-model is not working for latest version
        "huggingface-hub<=0.9.1",
        "sentencepiece",
        "scikit-learn",
        "datasets",
        "PySocks!=1.5.7",
        "charset-normalizer<3.0,>=2.0"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'relbert-train = relbert.relbert_cl.train:main',
            'relbert-eval-analogy = relbert.relbert_cl.evaluate:main_analogy',
            'relbert-eval-analogy-relation-data = relbert.relbert_cl.evaluate:main_analogy_relation_data',
            'relbert-eval-classification = relbert.relbert_cl.evaluate:main_classification',
            'relbert-eval-mapping = relbert.relbert_cl.evaluate:main_relation_mapping',
            'relbert-eval-loss = relbert.relbert_cl.evaluate:main_validation_loss',
            'relbert-push-to-hub = relbert.relbert_cl.push_to_hub:main'
        ]
    }
)

