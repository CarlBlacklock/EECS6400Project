import requests
import csv
import sys
from PIL import Image
from io import BytesIO


def getImages(sourceFileName, resultDir,startFrom=1, endAt=-1):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    sourceFile = open(sourceFileName, 'r', newline='', encoding='utf-8')
    logFile = open('{0}/log.txt'.format(resultDir), 'w')
    csv_reader = csv.DictReader(sourceFile)
    photos_saved = 0
    if endAt > 0:
        recordsRetrieved = 0
    #We may want to skip the first n records
    for i in range(1, startFrom):
        curr_record = next(csv_reader)
    for row in csv_reader:
        try:
            r = requests.get(row['Photo URL'], timeout=1.5)
            if r.status_code == requests.codes.ok and r.headers['content-type'] in image_formats:
                i = Image.open(BytesIO(r.content))
                i.save("{0}/{1}".format(resultDir, row['File name']))
                photos_saved += 1
        except:
            logFile.write('{0}\n'.format(row['Photo URL']))
        if endAt > 0:
            recordsRetrieved += 1
            if endAt == recordsRetrieved:
                break
    sourceFile.close()
    print(photos_saved)


if __name__ == '__main__':
    sourceFileName = sys.argv[1]
    resultDir = sys.argv[2]
    if len(sys.argv) == 4:
        print('Getting images starting at {0} entry'.format(sys.argv[3]))
        getImages(sourceFileName, resultDir, startFrom=int(sys.argv[3]))
    elif len(sys.argv) == 5:
        getImages(sourceFileName, resultDir, startFrom=int(sys.argv[3]), endAt=int(sys.argv[4]))
    else:
        getImages(sourceFileName, resultDir)