import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict


#attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}

#xmlfile = "000023.xml"


def generateVOC2Json(rootDir,xmlFiles,final_json):
	attrDict = dict()
	#images = dict()
	#images1 = list()
	attrDict["categories"]=[{"supercategory":"none","id":0,"name":"jj"},
			        {"supercategory":"none","id":1,"name":"st"},
			        {"supercategory":"none","id":2,"name":"dmyh_gh"},
			        {"supercategory":"none","id":3,"name":"lbj"}
			      ]
	images = list()
	annotations = list()
	for root, dirs, files in os.walk(rootDir):
		image_id = 0
		id1 = 1

		for file in xmlFiles:
			image_id = image_id + 1
			if file in files:
				
				#image_id = image_id + 1
				annotation_path = os.path.abspath(os.path.join(root, file))
				
				#tree = ET.parse(annotation_path)#.getroot()
				image = dict()
				#keyList = list()
				doc = xmltodict.parse(open(annotation_path).read())
				#print doc['annotation']['filename']
				image['file_name'] = str(doc['annotation']['filename'])
				#keyList.append("file_name")
				image['height'] = int(doc['annotation']['size']['height'])
				#keyList.append("height")
				image['width'] = int(doc['annotation']['size']['width'])
				#keyList.append("width")

				#image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
				image['id'] = image_id
				print( "File Name: {} and image_id {}".format(file, image_id))
				images.append(image)
				# keyList.append("id")
				# for k in keyList:
				# 	images1.append(images[k])
				# images2 = dict(zip(keyList, images1))
				# print images2
				#print images

				#attrDict["images"] = images

				#print attrDict
				#annotation = dict()
				# id1 = 1
				if 'object' in doc['annotation']:
					if type(doc['annotation']['object']) == OrderedDict:
						obj_list = [doc['annotation']['object']]
					else:
						obj_list = doc['annotation']['object']
					for obj in obj_list:
						# print(obj,obj_list)
						for value in attrDict["categories"]:
							annotation = dict()
							#if str(obj['name']) in value["name"]:
							if str(obj['name']) == value["name"]:
								#print str(obj['name'])
								# annotation["segmentation"] = []
								annotation["iscrowd"] = 0
								#annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
								annotation["image_id"] = image_id
								x1 = int(obj["bndbox"]["xmin"])  - 1
								y1 = int(obj["bndbox"]["ymin"]) - 1
								x2 = int(obj["bndbox"]["xmax"]) - x1
								y2 = int(obj["bndbox"]["ymax"]) - y1
								annotation["bbox"] = [x1, y1, x2, y2]
								annotation["area"] = float(x2 * y2)
								annotation["category_id"] = value["id"]
								annotation["ignore"] = 0
								annotation["id"] = id1
								annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
								id1 +=1

								annotations.append(annotation)
				
				else:
					print("File: {} doesn't have any object".format(file))
				#image_id = image_id + 1
				
			else:
				print("File: {} not found".format(file))
			

	attrDict["images"] = images	
	attrDict["annotations"] = annotations
	# attrDict["type"] = "segmentation"
	print('number of annotations:',len(annotations))

	#print attrDict
	jsonString = json.dumps(attrDict)
	with open(final_json, "w") as f:
		f.write(jsonString)


trainFile = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt"
final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_rgb/annotations/instances_train2017.json"
rootDir = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/Annotations/"
img_root = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/JPEGImages/"
coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_rgb/train2017/"

# trainFile = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt"
# final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_rgb/annotations/instances_val2017.json"
# rootDir = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/Annotations/"
# img_root = "/home/zexin/project/med_lung/datasets/train_voc_rgb/VOCdevkit2007/VOC2007/JPEGImages/"
# coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_rgb/val2017/"


trainFile = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt"
rootDir = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/Annotations/"
img_root = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/JPEGImages/"
final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_n1000_n200/annotations/instances_train2017.json"
coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_n1000_n200/train2017/"
## len 17985

trainFile = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt"
rootDir = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/Annotations/"
img_root = "/home/zexin/project/med_lung/datasets/voc_n1000_n200/VOCdevkit2007/VOC2007/JPEGImages/"
final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_n1000_n200/annotations/instances_val2017.json"
coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_n1000_n200/val2017/"
## len 4575

# trainFile = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt"
# rootDir = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/Annotations/"
# img_root = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/JPEGImages/"
# final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_n145_225/annotations/instances_train2017.json"
# coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_n145_225/train2017/"
# ## len 18000

# trainFile = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt"
# rootDir = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/Annotations/"
# img_root = "/home/zexin/project/med_lung/datasets/voc_n145_225/VOCdevkit2007/VOC2007/JPEGImages/"
# final_json = "/home/zexin/project/med_lung/datasets/coco_from_voc_n145_225/annotations/instances_val2017.json"
# coco_img_root = "/home/zexin/project/med_lung/datasets/coco_from_voc_n145_225/val2017/"
# ## len 4479
# os.mkdir(coco_img_root)

trainXMLFiles = list()
with open(trainFile, "r") as f:
	for line in f:
		fileName = line.strip()
		print(fileName)
		trainXMLFiles.append(fileName + ".xml")
		os.system('ln -s %s/%s.jpg %s'%(img_root,fileName,coco_img_root))

generateVOC2Json(rootDir, trainXMLFiles,final_json)




