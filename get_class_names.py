import argparse
import os

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

    for roots, dir, files in os.walk(root):
        print("类别总数量:  ", len(dir))
        return dir
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文件处理')
    parser.add_argument('--data_path', type=str, help='文件路径', default=r"../data/huamei/huamei-old/train/")  # need to do
    args = parser.parse_args()

    print(get_class_names(args))