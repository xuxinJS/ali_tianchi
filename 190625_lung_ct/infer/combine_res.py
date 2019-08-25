import argparse
import sys
sys.path.append('../lib')
from csvTools import writeCSV,readCSV

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--result_input1', default='1.csv', type=str)
    parser.add_argument('-i2', '--result_input2', default='2.csv', type=str)
    parser.add_argument('-o', '--result_output', default='all.csv', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    csv_path1 = args.result_input1
    csv_path2 = args.result_input2
    out_path = args.result_output
    all_csv = readCSV(csv_path1)
    csv_2 = readCSV(csv_path2)
    for i in csv_2[1:]:
        all_csv.append(i)
    writeCSV(out_path, all_csv)

if __name__ == '__main__':
    main()
