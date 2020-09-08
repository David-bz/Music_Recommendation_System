import os


def get_working_dir():
    sep = os.path.sep
    tokens = os.getcwd().split(sep)
    work_dir_idx = tokens.index('Music_Recommendation_System') + 1
    return os.path.join(os.path.abspath(os.sep), *tokens[:work_dir_idx]) + sep

def generate_path(linux_path):
    root_path = get_working_dir()
    tokens = linux_path.split('/')
    sep = "" if '.' in linux_path else os.path.sep
    return os.path.join(root_path, *tokens) + sep


