"""Splits a folder with the given format:
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        imgWhatever.jpg
        ...
    ...
into this resulting format:
    train/
        class1/
            img1.jpg
            ...
        class2/
            imga.jpg
            ...
    val/
        class1/
            img2.jpg
            ...
        class2/
            imgb.jpg
            ...
    test/
        class1/
            img3.jpg
            ...
        class2/
            imgc.jpg
            ...
"""

import pathlib
import random
import shutil
from os import path

try:
    from tqdm import tqdm

    tqdm_is_installed = True
except ImportError:
    tqdm_is_installed = False


def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]


def list_files(directory):
    """Returns all files in a given directory
    """
    return [
        f
        for f in pathlib.Path(directory).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]


def ratio(input, output="output", seed=1337, ratio=(0.8, 0.1, 0.1)):
    # make up for some impression
    assert round(sum(ratio), 5) == 1
    assert len(ratio) in (2, 3)

    if tqdm_is_installed:
        prog_bar = tqdm(desc=f"Copying files", unit=" files")

    for class_dir in list_dirs(input):
        split_class_dir_ratio(
            class_dir, output, ratio, seed, prog_bar if tqdm_is_installed else None
        )

    if tqdm_is_installed:
        prog_bar.close()

def ratio_first_name(input, output="output", seed=1337, ratio=(0.8, 0.1, 0.1)):
    # make up for some impression
    assert round(sum(ratio), 5) == 1
    assert len(ratio) in (2, 3)

    if tqdm_is_installed:
        prog_bar = tqdm(desc=f"Copying files", unit=" files")

    for class_dir in list_dirs(input):
        split_class_dir_ratio_first_name(
            class_dir, output, ratio, seed, prog_bar if tqdm_is_installed else None
        )

    if tqdm_is_installed:
        prog_bar.close()


def setup_files(class_dir, seed):
    """Returns shuffled files
    """
    # make sure its reproducible
    random.seed(seed)

    files = list_files(class_dir)

    files.sort()
    random.shuffle(files)
    return files


def setup_files_first_name(class_dir, seed):
    """Returns shuffled files via  26_20100527_062400.jpg  26
    """
    # make sure its reproducible
    first_name_set = set()
    files_names_list = list_files(class_dir)
    for i in files_names_list:
        first_name = i.name.split('_')[0]
        first_name_set.add(first_name)

    first_name_list = list(first_name_set)
    first_name_list_sort = sorted(first_name_list)
    random.seed(seed)
    random.shuffle(first_name_list_sort)
    return first_name_list_sort, files_names_list


def split_class_dir_ratio(class_dir, output, ratio, seed, prog_bar):
    """Splits one very class folder
    """
    files = setup_files(class_dir, seed)

    split_train = int(ratio[0] * len(files))
    split_val = split_train + int(ratio[1] * len(files))

    li = split_files(files, split_train, split_val, len(ratio) == 3)
    copy_files(li, class_dir, output, prog_bar)


def split_files(files, split_train, split_val, use_test):
    """Splits the files along the provided indices
    """
    files_train = files[:split_train]
    files_val = files[split_train:split_val] if use_test else files[split_train:]

    li = [(files_train, "train"), (files_val, "val")]

    # optional test folder
    if use_test:
        files_test = files[split_val:]
        li.append((files_test, "test"))
    return li


def split_class_dir_ratio_first_name(class_dir, output, ratio, seed, prog_bar):
    """Splits one very class folder via  26_20100527_062400.jpg  26
    """
    first_names, file_names = setup_files_first_name(class_dir, seed)

    split_train = int(ratio[0] * len(first_names))
    split_val = split_train + int(ratio[1] * len(first_names))

    li = split_files_first_name(first_names, file_names, split_train, split_val, len(ratio) == 3)
    copy_files(li, class_dir, output, prog_bar)


def split_files_first_name(first_names, file_names, split_train, split_val, use_test):
    """Splits the files along the provided indices  via  26_20100527_062400.jpg  26
    """
    first_names_train = first_names[:split_train]
    first_names_val = first_names[split_train:split_val] if use_test else first_names[split_train:]

    files_train = []
    files_val = []
    if use_test:
        files_test = []

    for i in file_names:
        first_name = i.name.split('_')[0]
        if first_name in first_names_train:
            files_train.append(i)
        elif first_name in first_names_val:
            files_val.append(i)
        else:
            if use_test:
                files_test.append(i)

    li = [(files_train, "train"), (files_val, "val")]

    # optional test folder
    if use_test:
        # files_test = files[split_val:]
        li.append((files_test, "test"))
    return li


def copy_files(files_type, class_dir, output, prog_bar):
    """Copies the files from the input folder to the output folder
    """
    # get the last part within the file
    class_name = path.split(class_dir)[1]
    for (files, folder_type) in files_type:
        full_path = path.join(output, folder_type, class_name)

        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        for f in files:
            if not prog_bar is None:
                prog_bar.update()
            shutil.copy2(f, full_path)


if __name__ == '__main__':
    input_folder = '/home/dls1/simple_data/classification/train/magnetogram'
    output_folder = '/home/dls1/simple_data/data_gen/0703_mag'
    ratio_first_name(input=input_folder, output=output_folder, seed=11, ratio=(0.6, 0.2, 0.2))
