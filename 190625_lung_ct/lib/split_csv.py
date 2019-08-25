"""
训练集的数据被分成train和test,将label也对应分成train和test
"""
import csv
import os

def writeCSV(filename, lines):
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


csv_path = '/HHD/xuxin/data/train_data/public/tianchi/190625_CT/dataset/chestCT_round1_annotation.csv'
train_folder = '/HHD/xuxin/data/train_data/public/tianchi/190625_CT/dataset/split/train'
test_folder = '/HHD/xuxin/data/train_data/public/tianchi/190625_CT/dataset/split/test'
label_train_path = '/HHD/xuxin/data/train_data/public/tianchi/190625_CT/dataset/split/train.csv'
label_test_path = '/HHD/xuxin/data/train_data/public/tianchi/190625_CT/dataset/split/test.csv'

train_list = []
test_list = []
for i in os.listdir(train_folder):
    i_strip = i.split('.')[0]
    if i_strip not in train_list:
        train_list.append(i_strip)
for i in os.listdir(test_folder):
    i_strip = i.split('.')[0]
    if i_strip not in test_list:
        test_list.append(i_strip)

train_list.sort(reverse=False)
test_list.sort(reverse=False)

label_all = readCSV(csv_path)
header = label_all[0]
label_train = []
label_test = []
label_train.append(header)
label_test.append(header)
for i in label_all[1:]:
    if i[header.index(header[0])] in train_list:
        label_train.append(i)
    if i[header.index(header[0])] in test_list:
        label_test.append(i)

writeCSV(label_train_path, label_train)
writeCSV(label_test_path, label_test)
