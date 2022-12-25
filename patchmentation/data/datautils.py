import os, shutil, tarfile, requests
from appdirs import user_cache_dir
from tqdm import tqdm

FOLDER_CACHE_NAME = 'patchmentation-data'
FOLDER_CACHE = user_cache_dir(FOLDER_CACHE_NAME)

def rm(path: str):
    print(f'remove {path}')
    if os.name == 'posix':
        _rm(path)
    else:
        _rmV2(path)

def _rm(path: str):
    os.system(f'rm -rf {path}')

def _rmV2(path: str):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def validate_not_exists(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and overwrite:
        rm(path)
    if os.path.exists(path):
        raise FileExistsError(path)

def download(url: str, file: str, overwrite: bool = False) -> None:
    print(f'download from {url} to {file} (overwrite={overwrite})')
    validate_not_exists(file, overwrite)
    os.makedirs(os.path.dirname(file), exist_ok=True)  
    if os.name == 'posix':  # check if os is linux distribution
        _download(url, file)
    else:
        _downloadV2(url, file)

def _download(url: str, file: str) -> None:
    os.system(f'wget -c {url} -O {file}')

def _downloadV2(url: str, file: str) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f'download {file}'):
                f.write(chunk)

def extract_tar(file: str, folder: str, overwrite: bool = False) -> None:
    print(f'extract tar from {file} to {folder} (overwrite={overwrite})')
    validate_not_exists(folder, overwrite)
    os.makedirs(folder)
    with tarfile.open(file) as f:
        f.extractall(folder)

def extract_zip(file: str, folder: str, overwrite: bool = False) -> None:
    print(f'extract zip from {file} to {folder} (overwrite={overwrite})')
    validate_not_exists(folder, overwrite)
    os.makedirs(folder)
    shutil.unpack_archive(file, folder, 'zip')

def remove_ext(file: str):
    return os.path.splitext(file)[0]