import os

def check_path(path):
    if not os.path.exists(path=path):
        os.mkdir(path=path)