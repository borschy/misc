import os
import time
import random
import shutil
import argparse


def gen_files(args):
    root = args.data_path
    files_list = []
    for roots, _, files in os.walk(root):
        for file in files:
            if file.endswith(args.postfix):
                files_list.append(os.path.join(roots, file))
    len_files = len(files_list)
    print("数据总数量:  ", len_files)
    return files_list


def gen_txt(args):
    files = gen_files(args)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='文件处理')
    parser.add_argument('--data_path', type=str, help='文件路径', default=r"data/drinks_dataset/")  # need to do
    parser.add_argument('--save_path', type=str, help='保存的路径', default=r"data/drinks_dataset/")  # need to do
    parser.add_argument('--postfix', type=str, help='要操作的文件后缀名', default=('.jpg','.jpeg'))
    parser.add_argument('--val', type=bool, help='是否要切割数据集', default=True)
    parser.add_argument('--ratio', type=str, help='训练集占比', default=0.4)
    args = parser.parse_args()
    print(type(args))
    start = time.time()
    gen_txt(args)
    print(f"Operation finished in {time.time()-start}")