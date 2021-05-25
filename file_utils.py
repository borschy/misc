import argparse
import random
import os
import time
import shutil


def gen_files(path, num_samples=None):
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
            if file.endswith(args.postfix):
                files_list.append(os.path.join(roots, file))
                if num_samples:
                    sample_count +=1
                    if sample_count >= num_samples: break

    len_files = len(files_list)
    print("数据总数量:  ", len_files)
    return files_list


def get_class_names(args):
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

    print("类别总数量:  ", len(folder_list))
    print(folder_list)


def paths2txt(args):
    ''' Get the paths of all the samples in train, val, test
        and write them to a file. Used in place of gen_train_val
        when the classes are already separated.
    Params:
        args (namespace): argparse arguments
    Returns:
        dict: 
            'train': train paths
            'val': val paths
    Author:
        Natasha
    Date:
        2021/05/20 
    '''
    sample_dict = {'train': '', 'val': ''}

    for phase in ['train','val']:
        folder = f"{args.data_path}/{phase}/"
        samples = gen_files(folder, args.num_samples)
        random.shuffle(samples)
        sample_dict[phase] = [sample + "\n" for sample in samples]

    if args.write:
        for phase in ['train','val']:
            if not os.path.exists(args.txt_path):
                os.mkdir(args.txt_path)
            with open(args.txt_path + f'/CFN-208_{phase}.txt', 'w') as f: # remember to edit this to be whichever project you want
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文件处理')
    parser.add_argument('--data_path', type=str, help='文件路径', default=r'../data/ChineseFoodNet-208')  # need to do NOTE: without trailing /
    parser.add_argument('--postfix', type=str, help='要操作的文件后缀名', default=('.jpg','.jpeg'))
    parser.add_argument('--txt_path', type=str, help='path to save the files to', default=r'../data/ChineseFoodNet-208/')
    parser.add_argument('--subset_path', type=str, help='path to save the subset to', default=r'../data/CFN-208_subset400')    
    parser.add_argument('--num_samples', type=int, help='要操作的文件后缀名', default=400)
    parser.add_argument('--write', type=bool, help='write to file', default=False)
    args = parser.parse_args()

    start = time.time()
    new_subset(args)
    print("Finished!")
    print(f"Operation completed in {time.time() - start}")
