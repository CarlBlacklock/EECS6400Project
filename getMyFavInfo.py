import flickrapi
import csv

key_file = open('flickr_api.txt', 'r')
email_file = open('email.txt', 'r')
csv_file = open('flickr_photos_list.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Photo URL', 'User Name', 'Real Name', 'Profile URL'])
email = email_file.readline()
email = email.rstrip('\n')
contents = key_file.readlines()
for i in range(len(contents)):
    contents[i] = contents[i].rstrip('\n')

key = contents[0].split()[1]
secret = contents[1].split[1]
key_file.close()
email_file.close()
flickr = flickrapi.FlickrAPI(key, secret)
me = flickr.people.findByEmail(find_email=email)
me_id = me.find('user').attrib['id']

my_favs = flickr.favorites.getList(user_id=me_id)
num_pages = int(my_favs.find('photo').attrib['pages'])
