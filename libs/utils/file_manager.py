import os
import pickle


def create_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        return

def read_file(file_name: str) -> list[str]:
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_name}")
    

def load_file_pickle(root_file: str, file_name: str):
    if not os.path.exists(root_file):
        create_folder(folder_name=root_file)

    file_name = os.path.join(root_file, file_name)
    if not os.path.exists(file_name):
        return None, None
    
    with open(file_name, 'rb') as f:
        reader = pickle.load(f)
    
    return reader['data_list'], reader['inputs']


def save_file_pickle(file_name: str, data: any):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)