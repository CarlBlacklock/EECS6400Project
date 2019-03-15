import os
import csv
import sys

path = 'C:/Users/cblacklo/Documents/EECS6400'
labelFile = open(os.path.join(path, 'labels.txt'), 'r')
csv_file = open(os.path.join(path, 'imageNames.csv'), 'w', newline='')
nameFile = open(os.path.join(path, 'images.txt'), 'w')
csv_writer = csv.writer(csv_file, delimiter=' ')
fileLine = labelFile.readline()
while not fileLine == '':
    if not fileLine[0] == '#':
        fileLine = fileLine.rstrip('\n')
        fileLine = fileLine.split('\t')
        temp = fileLine[0].split()
        if len(temp) > 1:
            file = temp[0]
            for i in range(1, len(temp)):
                file = "{0}_{1}".format(file, temp[i])
        else:
            file = temp[0]
        source = "lfw/{0}/{1}_{2}.jpg".format(file, file, str(fileLine[1]).zfill(4))
        file_name = '{0}_{1}.jpg'.format(file, str(fileLine[1]).zfill(4))
        csv_writer.writerow(file_name)
        nameFile.write('{0}\n'.format(file_name))
        os.rename(os.path.join(path, source), os.path.join(path, 'lfw/{0}'.format(file_name)))
    fileLine = labelFile.readline()
    
labelFile.close()
csv_file.close()
nameFile.close()
        
    

