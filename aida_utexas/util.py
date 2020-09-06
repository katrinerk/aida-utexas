import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Union


def get_input_path(path: Union[str, Path], check_exist: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if check_exist:
        assert path.exists(), f'{path} does not exist!'

    return path


def get_output_path(path: Union[str, Path], overwrite_warning: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.exists() and overwrite_warning:
        if not query_yes_no(f'{path} already exists, overwrite?', default='yes'):
            sys.exit(0)

    return path


def get_output_dir(dir_path: Union[str, Path], overwrite_warning: bool = True) -> Path:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if dir_path.is_dir() and any(True for _ in dir_path.iterdir()):
        if overwrite_warning:
            if not query_yes_no(f'{dir_path} already exists and is not empty, delete it first?',
                                default='yes'):
                sys.exit(0)
        shutil.rmtree(str(dir_path.resolve()))

    dir_path.mkdir(exist_ok=True, parents=True)

    return dir_path


def get_file_list(path: Union[str, Path], suffix: str = None, sort: bool = True,
                  ignore_hidden: bool = True) -> List[Path]:
    if isinstance(path, str):
        path = get_input_path(path)
    if path.is_file():
        file_list = [path]
    else:
        if suffix is None:
            file_list = [f for f in path.iterdir()]
        else:
            file_list = [f for f in path.glob('*{}'.format(suffix))]
        if sort:
            file_list = sorted(file_list)

    if ignore_hidden:
        file_list = [f for f in file_list if not f.name.startswith('.')]

    return file_list


def query_yes_no(question, default=None):
    """
    Ask a yes/no question via input() and return their answer.

    :param question: a string that is presented to the user
    :param default: the presumed answer if the user just hits <Enter>.
    It must be 'yes', 'no' or None (the default, meaning an answer is required of the user).
    :return: True or False
    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError(f'invalid default answer: {default}')

    while 1:
        print(question + prompt, end='')
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            print('Please respond with \'yes\' or \'no\' (or \'y\' or \'n\')')


def read_json_file(file_path: Union[str, Path], file_desc: str = None):
    file_path = get_input_path(file_path, check_exist=True)
    file_desc = file_desc or 'JSON object'
    logging.info(f'Reading {file_desc} from {file_path} ...')
    with open(str(file_path), 'r') as fin:
        return json.load(fin)
