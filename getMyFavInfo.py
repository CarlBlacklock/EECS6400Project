import flickrapi
import csv
import time
import webbrowser

key_file = open('flickr_api.txt', 'r')
email_file = open('email.txt', 'r')
csv_file = open('flickr_photos_list_full.csv', 'w', newline='', encoding='utf-8')
field_names = ['Photo URL', 'File name', 'File Type', 'User Name', 'Profile URL']
csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
csv_writer.writeheader()

email = email_file.readline()
email = email.rstrip('\n')
contents = key_file.readlines()
for i in range(len(contents)):
    contents[i] = contents[i].rstrip('\n')

key = contents[0].split()[1]
<<<<<<< HEAD
secret = contents[1].split()[1]
key_file.close()
email_file.close()
flickr = flickrapi.FlickrAPI(key, secret)
flickr.authenticate_via_browser(perms='read')
me = flickr.people.findByEmail(find_email=email)
me_id = me.find('user').attrib['id']

my_favs = flickr.favorites.getList(user_id=me_id, extras='url_o, original_format', per_page=500)
num_pages = int(my_favs.find('photos').attrib['pages'])
cur_page = 1
records_processed = 0
#We already have the first page and the Flickr API limits us to 3600 API calls per hour
while True:
    favs=my_favs.find('photos').findall('photo')
    print(len(favs))
    for i in range(len(favs)):
        cur_attrib = favs[i].attrib
        fileOwner = flickr.people.getInfo(user_id=cur_attrib['owner'])
        dict = {'Photo URL': cur_attrib['url_o']}
        dict['File name'] = '{0}_{1}_o.{2}'.format(cur_attrib['id'], cur_attrib['originalsecret'], cur_attrib['originalformat'])
        dict['File Type'] = cur_attrib['originalformat']
        dict['User Name'] = fileOwner.find('person').find('username').text
        dict['Profile URL'] = fileOwner.find('person').find('profileurl').text
        csv_writer.writerow(dict)
        records_processed += 1
        #To help ensure we don't go over the 3600 API calls per hour we impose a delay of 0.01 second between loop iterations
        time.sleep(0.1)
    print('Finished Favorite Page {0}'.format(cur_page))
    print('Processed {0} records'.format(records_processed))
    records_processed = 0
    cur_page += 1
    if cur_page > num_pages:
        break
    
    my_favs = flickr.favorites.getList(user_id=me_id, extras='url_o, original_format', page=cur_page, per_page=500)
    
csv_file.close()
