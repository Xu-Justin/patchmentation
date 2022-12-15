import os, shutil, tarfile
from appdirs import user_cache_dir

FOLDER_CACHE_NAME = 'patchmentation-data'
FOLDER_CACHE = user_cache_dir(FOLDER_CACHE_NAME)

def rm(path: str):
    print(f'remove {path}')
    os.system(f'rm -rf {path}')

def validate_not_exists(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and overwrite:
        rm(path)
    if os.path.exists(path):
        raise FileExistsError(path)

def download(url: str, file: str, overwrite: bool = False) -> None:
    print(f'download from {url} to {file} (overwrite={overwrite})')
    validate_not_exists(file, overwrite)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    os.system(f'wget -c {url} -O {file}')

def extract_tar(file: str, folder: str, overwrite: bool = False) -> None:
    print(f'extract tar from {file} to {folder} (overwrite={overwrite})')
    validate_not_exists(folder, overwrite)
    os.makedirs(folder, exist_ok=True)
    with tarfile.open(file) as f:
        f.extractall(folder)
