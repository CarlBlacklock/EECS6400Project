import csv
import ast
import random
import math
genderDict={'male':0, 'female':1}
ageDict = {'baby':0,'toddler':1, 'child':2, 'teen':3, 'adult':4, 'middle-aged':5,'senior':6}
def translateInput(rowInput):
    fileName = rowInput['External ID']
    labelDict = ast.literal_eval(rowInput['Label'])
    labels = [genderDict[labelDict['Gender'].lower()], ageDict[labelDict['Age'].lower()]]
    pair = (fileName, labels)
    return pair
    

sourceName = 'flickr_photos_labels.csv'
trainName = 'train_photo_labels.txt'
testName = 'test_photo_labels.txt'
sourceFile = open(sourceName, 'r', newline='')
trainFile = open(trainName, 'w',)
testFile = open(testName, 'w',)

sourceReader = csv.DictReader(sourceFile)
file_label_pairs = []
for row in sourceReader:
    file_label_pairs.append(translateInput(row))
sourceFile.close()    
train_set = random.sample(file_label_pairs, math.ceil(len(file_label_pairs)/3))
#test_set = list(set(file_label_pairs)-set(train_set))

for i in train_set:
    trainFile.write('{0}\t{1}\t{2}\n'.format(i[0], i[1][0], i[1][1]))
trainFile.close()
for i in file_label_pairs:
    if i not in train_set:
        testFile.write('{0}\t{1}\t{2}\n'.format(i[0], i[1][0], i[1][1]))
testFile.close()