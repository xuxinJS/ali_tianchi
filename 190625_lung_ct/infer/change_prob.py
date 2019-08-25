import sys
sys.path.append('../lib')
from csvTools import *

prob_thresh = 0.7
csv_file = 'ssd_resnet50_th0p3.csv'
save_file = 'out.csv'
source_result = readCSV(csv_file)
header = source_result[0]
out_result = []
out_result.append(header)
for i in source_result[1:]:
    # print(i[5])
    if float(i[5]) >= prob_thresh:
        i[5] = 1.0
    # else:
    #     i[5] = float(i[5]) * 1.5
    out_result.append(i)
writeCSV(save_file, out_result)