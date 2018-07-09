import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='timebox',
    version='0.0.5',
    author='Brian Kopp',
    author_email='briankopp.usa@gmail.com',
    description='An IO library for pandas',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/briankopp/timebox',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'python-dateutil',
        'pytz',
        'six'
    ],
    classifier=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Ubuntu'
    ),
)
