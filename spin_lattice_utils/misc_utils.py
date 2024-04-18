# %%
import os
import shutil

def create_path(path, bk=True):
    """create 'path' directory. If 'path' already exists, then check 'bk':
       if 'bk' is True, backup original directory and create new directory naming 'path';
       if 'bk' is False, do nothing.

    Args:
        path ('str' or 'os.path'): The direcotry you are making.
        bk (bool, optional): If . Defaults to False.
    """
    path += '/'
    if os.path.isdir(path):
        if bk:
            dirname = os.path.dirname(path)
            counter = 0
            while True:
                bkdirname = dirname + ".bk{0:03d}".format(counter)
                if not os.path.isdir(bkdirname):
                    shutil.move(dirname, bkdirname)
                    break
                counter += 1
            os.makedirs(path)
            # print("Target path '{0}' exsists. Backup this path to '{1}'.".format(path, bkdirname))
        else:
            None
            # print("Target path '{0}' exsists. No backup for this path.".format(path))
    else:
        os.makedirs(path)