from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
    print(long_description)

setup(
    name='MolFeatures',
    version='0.9009',
    packages=find_packages(),
    description='Your package description',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Eden Specktor',
    author_email='edenpsec@post.bgu.ac.il',
    url='https://github.com/edenspec2/Automation_code-main',
    install_requires=[],
    package_data={
        # If 'feather_example' is a Python package with __init__.py
        'MolFeatures': ['feather_example/*','Workshop_Example_Data/*','pictures/*', 'description.txt', 'README.md', 'requirements.txt','setup.py','usage_doc.ipynb'],
    },
    include_package_data=True,
)