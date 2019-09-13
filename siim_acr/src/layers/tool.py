# from common import *
import os
import shutil
import builtins
# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


def write_list_to_file(strings, list_file):
    with open(list_file, 'w') as f:
        for s in strings:
            f.write('%s\n'%s)
    pass


# backup ------------------------------------
#https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
def backup_project_as_zip(project_dir, zip_file):
    shutil.make_archive(zip_file.replace('.zip',''), 'zip', project_dir)
    pass

# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    # if type(optimizer)==optim.SGD or (type(optimizer)==optim.RMSprop and lr<0.0001):
    #     # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[param_group['lr']]
    return lr

def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))