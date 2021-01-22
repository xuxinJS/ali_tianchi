import os
import json

resp=[]
with open("/home/yuyu/APPS/repo/yolov5/runs/detect/exp5/labels/save.json","r") as f:
    a=json.loads(f.read())
    for a_i in a:
        a_i["category"]=a_i["category"]+1
        resp.append(a_i)

with open("./test.json","w") as f:
    f.write(json.dumps(resp))