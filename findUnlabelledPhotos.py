import os
import csv
import sys

path = './Flickr_Photos_Full'
path_new = './Flickr_Photos_Full/unlabelled'
label_file = open('flickr_photos_labels.csv', 'r', newline='')
source_file = open('flickr_photos_list_full.csv', 'r', newline='', encoding='utf-8')
label_reader = csv.DictReader(label_file)
source_reader = csv.DictReader(source_file)

label_list = []

for row in label_reader:
    label_list.append(row['External ID'])
    
for row in source_reader:
    if not row['File name'] in label_list:
        os.rename(os.path.join(path, row['File name']), os.path.join(path_new, row['File name']))
        
label_file.close()
source_file.close()