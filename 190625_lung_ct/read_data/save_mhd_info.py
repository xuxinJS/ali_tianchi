import sys
from glob import glob
import os

sys.path.append('../lib')
from preprocess import load_itk
from csvTools import writeCSV

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input folder", type=str,
                        default="/T3/data/train_data/public/tianchi/190625_CT/dataset/split/test")
    parser.add_argument("-o", "--output_csv", help="output folder", type=str,
                        default="real_test.csv")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_dir = args.input_folder
    save_path  = args.output_csv
    file_list = glob(data_dir + "/*.mhd")
    csv_list = []
    header = ['seriesuid', 'origin_x', 'origin_y', 'origin_z', 'spacing_x', 'spacing_y', 'spacing_z']
    csv_list.append(header)
    for file_name in file_list:
        file_id = os.path.basename(file_name).split('.')[0]
        ct_scan, origin, spacing = load_itk(file_name)
        csv_list.append([file_id, origin[2], origin[1], origin[0], spacing[2], spacing[1], spacing[0]])
        print(file_id, origin[2], origin[1], origin[0], spacing[2], spacing[1], spacing[0])
    # print(csv_list)
    writeCSV(save_path, csv_list)


if __name__ == '__main__':
    main()
