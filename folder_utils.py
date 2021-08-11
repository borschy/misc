# -- coding: utf-8 --

import os
import time
import random
import shutil
import argparse
import re
import json
import pickle

import cv2
import codecs
from tqdm import tqdm
from torchvision.datasets import ImageFolder


def gen_txt(args):
    files = get_files(args)
    random.shuffle(files)

    files = [file + "\n" for file in files]
    if args.val:
        trains = files[:int(len(files) * args.ratio)]
        vals = files[int(len(files) * args.ratio):]
        # vals = files[int(len(files) * args.ratio):int(len(files) * args.ratio*2)]
        # test = files[int(len(files) * args.ratio*2):]
    else:
        trains = files
        vals = files
        # test = files

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(args.save_path + r'/train.txt', 'w') as f:
        f.writelines(trains)

    with open(args.save_path + r'/val.txt', 'w') as f:
        f.writelines(vals)

    # with open(args.save_path + r'/test.txt', 'w') as f:
    #     f.writelines(test)


def get_class_names(root):
    ''' Get the folder names of each class.
    (Folder names usually correspond to class names or ids)
    Params:
        args (namespace): argparse arguments
    Returns:
        list: list of class names
    Author:
        Natasha (documentation)
    Date:
        2021/05/19
    '''
    folder_list = os.listdir(root) # folder names are class names
    '''
    for i in os.listdir(root):
        folder_list.append(i)'''
    return folder_list[1:]


def gen_yolo_lbl(args):
    ''' Generate label txt files in yolo style.
    Labels are based on idx, not folder name
    [class, x_center, y_center, w, h].
    Takes in a structured folder heirarchy where all images in the 
    same folder belong to the same class.
        Params:
            args (namespace): data path from argparse
        Returns:
            None
        Author: 
            Jay
        Date: 
            2021/05/19
    '''
    root = args.data_path
    num_files = get_files(args.data_path)

    folder_list = get_ds_mapping(args)
    print("Making labels!")
    with tqdm(range(len(num_files))) as pbar:

        for roots, dir, files in os.walk(root):

            for file in files:
                if file.endswith((".jpg", ".jpeg")):
                    file = os.path.join(roots, file)

                    file_pathname, _ = os.path.splitext(file)
                    txt_file = file_pathname + ".txt"
                    folder_name = os.path.basename(roots)
                    folder_index = folder_list.index(folder_name)
                    with open(txt_file, "w") as f:

                        # we only need folder_index instead of -1 because get_class_names() already accounts for root
                        f.write(str(folder_index) + " " + str(0.5) + " " + str(0.5) + " " + str(1) + " " + str(1))
                pbar.update(1)
    print("Finished making labels!")


def make_lbl(root):
    for roots, dir, files in os.walk(root):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                file = os.path.join(roots, file)
                file_pathname, ext = os.path.splitext(file)
                txt_file = file_pathname + ".txt"
                lbl = roots[-2:]
                with open(txt_file, "w") as f:
                    f.write(lbl + " " + str(0.5) + " " + str(0.5) + " " + str(1) + " " + str(1))


def cfn_test_lbl_to_yolo(args):
    # read from CFNs test_truth_list file to get class names
    txt_file = args.data_path+"/test_truth_list.txt"
    test_path = args.data_path+"/test/"
    imgs_and_lbls = dict()

    with open(txt_file, "r") as f:
        lines = f.readlines()
        for string in lines:
            file_name, lbl = string.split(" ")
            imgs_and_lbls[file_name] = lbl[:-1]

    for x in imgs_and_lbls:
        idx = x.split(".")[0]
        with open(f"{test_path}{idx}.txt", "w") as f:
            f.write(imgs_and_lbls[x] + " " + str(0.5) + " " + str(0.5) + " " + str(1) + " " + str(1))


def get_files(path, num_samples=None):
    ''' Get the paths of the image files.
    Params:
        path (str): path of the root folder
        num_samples (int): number of samples per class
    Returns:
        list: list of file paths
    Author:
        Natasha (documentation)
    Date:
        2021/05/19
    '''
    files_list = []
    for roots, _, files in os.walk(path):
        sample_count = 0
        if num_samples:
            random.shuffle(files)
        for file in files:
            if file.endswith((".jpeg", ".jpg", ".png")):
                files_list.append(os.path.join(roots, file))
                if num_samples:
                    sample_count +=1
                    if sample_count >= num_samples: break

    len_files = len(files_list)
    # print("数据总数量:  ", len_files)
    return files_list


def get_files_by_class(path, num_samples=None):
    ''' Get the paths of the image files BY CLASS.
        Used to ensure the train/test split is split by 
        class, not just randomly.
    Params:
        path (str): path of the root folder
        num_samples (int): number of samples per class
    Returns:
        dict: list of file paths for each class
    Author:
        Natasha (documentation)
    Date:
        2021/07/12
    '''

    dirs = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]
    files_dict = {folder:[] for folder in dirs}
    for dir in dirs:
        files = [os.path.join(path,dir,file) for file in os.listdir(os.path.join(path,dir)) if os.path.isfile(os.path.join(path,dir,file))]
        sample_count = 0
        if num_samples:
            random.shuffle(files)
        for file in files:
            if file.endswith((".jpeg", ".jpg")):
                files_dict[dir].append(file)
                if num_samples:
                    sample_count +=1
                    if sample_count >= num_samples: break

    len_files = len(files_dict)
    # print("数据总数量:  ", len_files)
    return files_dict


def get_class_names(args, include_index=False):
    ''' Get the folder names of each class.
    Params:
        args (namespace): argparse arguments
    Returns:
        list: list of class names
    Author:
        Natasha (documentation)
    Date:
        2021/05/19
    '''
    root = args.data_path
    folder_list = []
    for roots, _, files in os.walk(root):
        class_name = os.path.split(roots)[-1]
        folder_list.append(class_name)
    
    mod = sorted(folder_list[1:])
    if include_index:
        food_dict = {}
        for idx,cls in enumerate(mod):
            food_dict[cls] = str(idx)
        return food_dict
    else: return folder_list[1:]


def paths2txt(args, dataset_name=''):
    ''' Get the paths of all the samples in train, val, test
        and write them to a file. Used in place of gen_train_val
        when the classes are already separated.
    Params:
        args (namespace): argparse arguments
        dataset_name (str): dataset name to name the txt file with
    Returns:
        dict: 
            'train': list of train paths
            'val': list of val paths
    Author:
        Natasha
    Date:
        2021/05/20 
    '''
    sample_dict = {'train': '', 'val': '', 'test':''}

    for phase in ['train', 'test']:
        folder = os.path.join(args.data_path, phase)
        samples = get_files(folder)
        random.shuffle(samples)
        sample_dict[phase] = [sample + "\n" for sample in samples]

    if False:
        for phase in ['train','val']:
            if not os.path.exists(args.txt_path):
                os.mkdir(args.txt_path)
            with open(args.txt_path + f'/{dataset_name}_{phase}.txt', 'w') as f: # remember to edit this to be whichever project you want
                f.writelines(sample_dict[phase])
    return sample_dict


def new_subset(args):
    sample_dict = paths2txt(args)

    for phase in ['train', 'val']:
        for sample in sample_dict[phase]:
            _, _, _, stage, lbl, filename = sample.split('/')
            path = sample[:-4] # without extention and \n
            copy_dir = f"{args.subset_path}/{stage}/{lbl}/"

            if not os.path.exists(copy_dir):
                os.makedirs(copy_dir)

            for filetype in ['jpg', 'txt']:
                save_path = f"{copy_dir}{filename[:-4]}{filetype}"
                shutil.copy(path+filetype, save_path)


def move_single_img(roots):
    ''' Remove a single image from the child folder to the parent folder.
        Params: 
            --data_dir: path of the parent directory
        Returns:
            Null
        Author:
            Natasha
        Date:
            2021/05/12
    '''
    # os.walk is already recursive so this is wrong
    for folder in os.listdir(roots): # parent folder
        cls_fldr = os.path.join(roots, folder) # class folder
        files = os.listdir(cls_fldr)
        if len(files) == 0:
            print(cls_fldr)
                    # move to parent folder
                # if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    # imgs.append(os.path.join(root, file))
                    # print(f"Child: {child},\n File: {file}\n\n")
                    # shutil.move(child+"\\"+file, parent+"\\"+file)
            os.rmdir(cls_fldr)
            continue
    # return imgs


def merge_small_datasets(root, save):
    ''' Move images from sub_cls directory to big_cls directory.
        Params: 
            root
            save
        Returns:
            Null
        Author:
            Natasha
        Date:
            2021/07/21
    '''
    num_files = len(get_files(root))
    with tqdm(range(num_files)) as pbar:
        sub_classes = [os.path.join(root,folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root,folder))]
        for folder in sub_classes:
            files = get_files(folder)
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    src = os.path.join(folder, file)
                    dst = os.path.join(save, os.path.split(file)[1])
                    # if os.path.exists(dst): continue
                    # print(f"src {src},\n dst: {dst}\n\n")
                    try:
                        shutil.copy(src, dst)
                    except FileExistsError as e:
                        continue
                    except shutil.SameFileError as e:
                        continue
                    except OSError as e:
                        os.makedirs(save)

                        shutil.copy(src, dst)
                    pbar.update(1)


def rename_cls_folders(args, cls_list=None):
    ''' Rename the folder to the folder index.
        Params: 
            args (namespace): uses the --data_path
        Returns:
            Null
        Author:
            Natasha
        Date:
            2021/06/09
    '''
    counter = 0
    for path, dir, files in os.walk(args.data_path):
        if len(files)==0: continue
        if cls_list:
            dst = os.path.join(args.data_path, cls_list[counter])
        else:
            dst = os.path.join(args.data_path, f"{counter:02d}")
        os.rename(path, dst)
        # print("src: " + path + "\tdst: " + dst + "\n\n")
        counter+=1


def rename_folders_from_txt(args):
    ''' Rename the folders according to each folder's .txt labels.
    '''
    for path, dir, files in os.walk(args.data_path):
        dir.reverse()
        if len(files)==0: continue

        txt_file = os.path.join(path,files[1])

        with open(txt_file, "r") as f:
            lbl = int(f.read()[:2])
            print(lbl)
        dst = os.path.join(args.data_path, lbl)
        print(dst)
        os.rename(path, dst)
       

def split_dataset(args, root=None, save=None, ratio=None, move_txt=False):
    ''' Moves both the labels and samples.
    '''
    if not root:
        root = args.data_path
        save = args.save_path
        ratio = args.ratio
    
    files = get_files_by_class(root) # dict of all the samples in each class
    for key in files:
        random.shuffle(files[key])

    sample_dict = {"train":[], "val":[], "test":[]}
    
    for key in files:
        sample_dict["train"].extend(file for file in files[key][:int(len(files[key]) * ratio[0])])
        sample_dict["val"].extend(file for file in files[key][int(len(files[key]) * ratio[0]):-int(len(files[key]) * ratio[2])])
        sample_dict["test"].extend(file for file in files[key][-int(len(files[key]) * ratio[2]):])

    print("Splitting dataset!")
    for phase in ["train", "val", "test"]: 
        with tqdm(range(len(sample_dict[phase]))) as pbar:
            print(f"{phase}: {len(phase)}")
            for sample in sample_dict[phase]:
                path = re.split("\\\\|/", sample)
                cls, file = path[-2:]
                cls_folder = os.path.join(save, phase, cls)
                if not os.path.exists(cls_folder): os.makedirs(cls_folder)

                if move_txt:
                    for filetype in [".jpg",".txt"]:
                        file_name, ext = os.path.splitext(sample)
                        if ext == ".jpeg" and filetype == ".jpg": filetype = ".jpeg"
                        file_name = file_name + filetype
                        
                        dst = os.path.join(cls_folder, os.path.basename(file_name))
                        shutil.copy(file_name, dst)
                else:
                    dst = os.path.join(cls_folder, os.path.basename(sample))
                    try:
                        shutil.copy(sample, dst)
                    except FileExistsError as e:
                        continue
                    except shutil.SameFileError as e:
                        continue
                pbar.update(1)
    print("Finished splitting dataset!")


def get_ds_stats(args):
    # TODO: finish
    ''' Make a txt file of class names and number of samples.
    Params:
        args (namespace): 
            args.data_path: path of the dataset folder
            args.save_path: path to save the txt (and processed data later)
    Returns:
        Null
    Author:
        Natasha
    Date:
        2021/06/15
    '''
    names = get_class_names(args) # list
    # TODO: make a func to get the # of samples per class
    # OR just modify names to also get the # of images
    name_num = {}

    for name in names:
        name_num[name] = None
    for i, (k,v) in enumerate(name_num): pass

    # write to file
    if not os.path.exists(args.save_path):
        os.mkdir(args.txt_path)
    with open(args.txt_path + f'dataset_stats.txt', 'w') as f: # remember to edit this to be whichever project you want
        f.writelines(name_num)


def get_num_samples(args, path=False, display=True):
    if not path:
        path = args.data_path
    lbl_num = dict()

    for root, dirs, files in os.walk(path):
        for directory in dirs:
            class_folder = os.path.join(root, directory)
            lbl_num[directory] = len(get_files(class_folder))
        break    

    # new = {k: v for k, v in lbl_num.items() if v > 75}
    new = {k: v for k, v in lbl_num.items()}
    
    if display:
        str_rep = str(new).replace(",","\n")
        print(str_rep + "\n" + str(len(new)))
    return new
    

def rename_folders_from_txt2(args):
    ''' Rename folders from their indices to their names based on 
        a single text file containing folder indexes and corresponding names.
    '''
    with open(args.txt, "r", encoding='utf-8') as f:
        lbls_names = {}
        for line in f:
            splitlines = line.split(" ")
            idx = splitlines[0]

            name = splitlines[1:-1]
            if len(name) > 1: 
                for word in name: 
                    try: 
                        int(name)
                        name.remove(word)
                    except Exception: pass
                name = name[0] + "_" + name[1]
            else: name = name[0]
            lbls_names[idx] = name

        for path, dir, files in os.walk(args.data_path):
            lbl = path.split("\\")[-1]
            if lbl in args.data_path: continue
            dst = os.path.join(args.data_path, lbls_names[lbl])
            # print(f"path:\t{path}")
            # print(f"dst:\t{dst}")
            os.rename(path, dst)
        print(lbls_names.values())


def set_cls_mapping(mapping):
    '''{C0: [c0, c1, ...],
        C1: [c0, c1, ...],
        ...
        Cn: [c0, c1, ...]}  ----> pkl
    '''
    with open('cls_name_idx.pkl', 'wb') as p_f:
        pickle.dump(mapping, p_f)


def get_cls_mapping(file='cls_name_idx.pkl'):
    with open(file, 'rb') as p_f:
        data = pickle.load(p_f)
    return data


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(dst): os.makedirs(dst)
            shutil.copy2(s, d)


def copy_folders_from_dict(args, source=False, dest=False):

    folders = get_num_samples(source if source else args, display=False)
    for dir in os.listdir(source if source else args.data_path):
        if dir in folders.keys():
            src = os.path.join(source if source else args.data_path, dir)
            dst = os.path.join(dest if dest else args.save_path, dir)
            if os.path.exists(dst): continue
            else: copytree(src,dst)


def get_ds_mapping(args=None, data_path=False):
    dataset = ImageFolder(data_path if data_path else args.data_path)
    # print(dataset.classes)
    return dataset.classes


def fix_minshi_nesting(args):
    folders = os.listdir(args.data_path)
    with tqdm(range(len(folders))) as pbar:
        for dir in folders:
            cls_folder = os.path.join(args.data_path, dir)
            for subdir in os.listdir(cls_folder):
                if "敏实" in subdir:
                    sub_folder = os.path.join(cls_folder, subdir)
                    if os.path.isdir(sub_folder):
                        files = get_files(sub_folder)

                        # if there's nothing in the subfolder
                        if len(files) == 0: 
                            os.rmdir(sub_folder)
                            continue

                        for file in files:
                            file_name = os.path.split(file)[-1]
                            # print(f"file: {file} \n cls_folder: {os.path.join(cls_folder, file_name)}\n\n")
                            shutil.move(file, os.path.join(cls_folder, file_name))
                        # after removing all samples, delete the folder
                        os.rmdir(sub_folder)
            pbar.update(1)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='文件处理')
    parser.add_argument('--data_path', type=str, help='文件路径', default=r"F:\data\detection\detection_split\images")  # need to do
    parser.add_argument('--save_path', type=str, help='保存的路径', default=r"X:\\InnoTech_staff\\Jay\\Data\\food_big_class_split")  # need to do
    parser.add_argument('--txt', type=str, help='文件路径', default=r"F:\data\detection\detection_split")  # need to do
    parser.add_argument('--postfix', type=str, help='要操作的文件后缀名', default=('.jpg','.jpeg'))
    parser.add_argument('--val', type=bool, help='是否要切割数据集', default=True)
    parser.add_argument('--ratio', type=list, help='训练集占比', default=[0.6, 0.3, 0.1]) # NOTE: what percentage goes in TRAIN 
    args = parser.parse_args()
    start = time.time()

    # print(paths2txt(args))
    
    annotations = {'train': ['F:\\data\\detection\\detection_split\\annotations\\train\\1a53c7adc727487cac36e8be3be8ff3e.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\0563ff3261084cfb96a5d3989d380ce4.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\6e76def3797a4cc0b0beb01d05e8e1d1.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\69fba825375147d28bbe79c31ba756a5.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\5ceed83e4f1d4ee08d6dc77cc492d2ca.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\10dc9c68d3a54be7b81ea2d03bd54a4f.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\19d5eed94765434fa3792efbbd817fc8.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\46758585ba7d4449b892126bd6851628.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\63020b7202114419acfc39f4ac36fd0c.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\439c8948488e481fa1612aa6eb5d1f92.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\835a29ffedd249619131a1e45b77df42.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\772b19ac76f74e4789099f53ecf90d44.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\4621aed02c4f4517b50944aef54a3878.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\8d81b16c88404022a41059da33dc848c.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\98c09e27bc054a778daeae14fd6e4a76.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\46dda58c96e543578bcf13fc7149a1d2.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\29da3b2b0c9441fa9c0cb38974486931.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\65f12195fe0c49c9a53c5550acc62648.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\15480b5509e24568b65a55d9835d3fa9.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\8ac3d7eb10b344e2a4ef5f29c66ff086.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\15dcd520d2594aa3a612e38cd669cab6.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\24cbc11e9e164151897ca8820f31d6dc.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\51ee9c70a7514c5a9f8f111a05d3fcbf.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\20dc946f1a3d48259cf1e291074789dd.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\3e438663624749638d4dfd80e7df1ca3.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\3c559886fcf7453da183f9ca888e0d6c.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\6709c3ed3d5d475ea5114a6ba18661fa.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\4938a8db914d48d793ff50b623d25f65.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\1e3a91cdf9e24af4a84f5cbc3dcd5a43.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\3c90d8d12cd84e20adac2f4b14e3577f.png\n', 'F:\\data\\detection\\detection_split\\annotations\\train\\98cc25897acd42ad83b3604f9052869c.png\n'], 'val': '', 'test': ['F:\\data\\detection\\detection_split\\annotations\\test\\1f02dc4a6fff4acdafe82d7d3930f508.png\n', 'F:\\data\\detection\\detection_split\\annotations\\test\\2b92c9aaaf5948918b52f3d715631364.png\n', 'F:\\data\\detection\\detection_split\\annotations\\test\\6cce9715020d4195b27acff03ab73316.png\n', 'F:\\data\\detection\\detection_split\\annotations\\test\\5f42346b643b48f59b122e701f8d6718.png\n', 'F:\\data\\detection\\detection_split\\annotations\\test\\3c1f91dfe173437a829124266f8d9510.png\n', 'F:\\data\\detection\\detection_split\\annotations\\test\\2a0a31b92aba436cab67817fbd70060e.png\n']}
    images = {'train': ['F:\\data\\detection\\detection_split\\images\\train\\3e438663624749638d4dfd80e7df1ca3.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\69fba825375147d28bbe79c31ba756a5.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\24cbc11e9e164151897ca8820f31d6dc.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\772b19ac76f74e4789099f53ecf90d44.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\29da3b2b0c9441fa9c0cb38974486931.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\98c09e27bc054a778daeae14fd6e4a76.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\3c90d8d12cd84e20adac2f4b14e3577f.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\98cc25897acd42ad83b3604f9052869c.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\4621aed02c4f4517b50944aef54a3878.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\65f12195fe0c49c9a53c5550acc62648.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\63020b7202114419acfc39f4ac36fd0c.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\19d5eed94765434fa3792efbbd817fc8.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\15dcd520d2594aa3a612e38cd669cab6.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\5ceed83e4f1d4ee08d6dc77cc492d2ca.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\20dc946f1a3d48259cf1e291074789dd.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\6709c3ed3d5d475ea5114a6ba18661fa.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\1a53c7adc727487cac36e8be3be8ff3e.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\46758585ba7d4449b892126bd6851628.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\1e3a91cdf9e24af4a84f5cbc3dcd5a43.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\835a29ffedd249619131a1e45b77df42.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\8d81b16c88404022a41059da33dc848c.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\51ee9c70a7514c5a9f8f111a05d3fcbf.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\46dda58c96e543578bcf13fc7149a1d2.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\8ac3d7eb10b344e2a4ef5f29c66ff086.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\3c559886fcf7453da183f9ca888e0d6c.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\6e76def3797a4cc0b0beb01d05e8e1d1.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\4938a8db914d48d793ff50b623d25f65.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\15480b5509e24568b65a55d9835d3fa9.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\0563ff3261084cfb96a5d3989d380ce4.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\439c8948488e481fa1612aa6eb5d1f92.jpg\n', 'F:\\data\\detection\\detection_split\\images\\train\\10dc9c68d3a54be7b81ea2d03bd54a4f.jpg\n'], 'val': '', 'test': ['F:\\data\\detection\\detection_split\\images\\test\\2b92c9aaaf5948918b52f3d715631364.jpg\n', 'F:\\data\\detection\\detection_split\\images\\test\\3c1f91dfe173437a829124266f8d9510.jpg\n', 'F:\\data\\detection\\detection_split\\images\\test\\2a0a31b92aba436cab67817fbd70060e.jpg\n', 'F:\\data\\detection\\detection_split\\images\\test\\5f42346b643b48f59b122e701f8d6718.jpg\n', 'F:\\data\\detection\\detection_split\\images\\test\\1f02dc4a6fff4acdafe82d7d3930f508.jpg\n', 'F:\\data\\detection\\detection_split\\images\\test\\6cce9715020d4195b27acff03ab73316.jpg\n']}
    final = {'train':[], 'test':[]}
    for phase in final:
        annotations[phase].sort()
        images[phase].sort()
        for i in range(len(annotations[phase])):
            assert len(annotations[phase]) == len(images[phase])
            final[phase].append(images[phase][i].strip() + " " + annotations[phase][i])
    
        with open(f'{args.txt}/{phase}.txt', 'w') as f: # remember to edit this to be whichever project you want
            f.writelines(final[phase])

    print(f"Operation finished in {time.time()-start}")