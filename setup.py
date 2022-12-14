import setuptools

version = '0.1.5'

def parse_version(version):
    _version = version.split('.')
    assert len(_version) == 3
    assert _version[0].isdigit()
    assert _version[1].isdigit()
    assert _version[2].isdigit()
    return version

def read(file):
    with open(file, 'r') as f:
        return f.read()

# ----- DON'T CHANGE THE CONFIGURATIONS BELOW -----

NAME = 'patchmentation'
VERSION = parse_version(version)

DESCRIPTION = 'A python library to perform patch augmentation'
LONG_DESCRIPTION = read('README.md')
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'

AUTHOR = 'William Justin, JonathanTho'
AUTHOR_EMAIL = 'williamjustinxu@gmail.com'
LICENSE_FILES = ['LICENSE']

URL = 'https://github.com/Xu-Justin/patchmentation'
PROJECT_URLS = {
    'Bug Tracker'  : 'https://github.com/Xu-Justin/patchmentation/issues',
    'Documentation': '',
    'Source Code'  : URL,
}
PACKAGES = setuptools.find_packages(include=[
    'patchmentation', 
    'patchmentation.*'
])
INSTALL_REQUIRES = [
    'matplotlib>=3.5.3',
    'numpy>=1.23.1',
    'opencv-python>=4.6.0.66',
    'typing-extensions>=4.3.0',
    'scipy>=1.9.3',
    'appdirs==1.4.4',
    'wget==3.2',
    'tqdm'
]
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
]

# ----- END OF CONFIGURATIONS -----

def setup(**kwargs):
    setuptools.setup(**kwargs)

if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        packages=PACKAGES,
        license_files=LICENSE_FILES,
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS
    )
