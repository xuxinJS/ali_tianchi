import sys
from glob import glob
import os

sys.path.append('../lib')
from preprocess import load_itk
from csvTools import readCSV

annos = readCSV('/home/admin/jupyter/Demo/DataSets/split/test.csv')
id_list = []
for i in annos[1:]:
    if i[0] not in id_list:
        id_list.append(i[0])

print(sorted(id_list))
print(len(id_list))

