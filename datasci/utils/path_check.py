import os


def check_path(path, is_make=True):
    if not os.path.exists(path=path):
        if is_make:
            os.mkdir(path=path)
        else:
            raise FileNotFoundError("Path %s is not Found" % path)