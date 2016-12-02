import json
import os
from pprint import pprint

with open('./yelp-dataset/photo_id_to_business_id.json') as data_file:
    data = json.load(data_file)

f = 0
d = 0
i = 0
m = 0
o = 0

os.system("mkdir -p ./yelp-photos/Drink/")
os.system("mkdir -p ./yelp-photos/Food/")
os.system("mkdir -p ./yelp-photos/Menu/")
os.system("mkdir -p ./yelp-photos/Inside/")
os.system("mkdir -p ./yelp-photos/Outside/")

for key in data:
    if key["label"] == 'drink':
        os.system("mv ./yelp-photos/" + key["photo_id"] + ".jpg ./yelp-photos/Drink/")
        d+=1
    if key["label"] == 'food':
        os.system("mv ./yelp-photos/" + key["photo_id"] + ".jpg ./yelp-photos/Food/")
        f+=1
    if key["label"] == 'menu':
        os.system("mv ./yelp-photos/" + key["photo_id"] + ".jpg ./yelp-photos/Menu/")
        m+=1
    if key["label"] == 'inside':
        os.system("mv ./yelp-photos/" + key["photo_id"] + ".jpg ./yelp-photos/Inside/")
        i+=1
    if key["label"] == 'outside':
        os.system("mv ./yelp-photos/" + key["photo_id"] + ".jpg ./yelp-photos/Outside/")
        o+=1
print ("Food = " + str(f) + "\nDrink = " + str(d) + "\nMenu = " + str(m) + "\nInside = " + str(i) + "\nOutside = " + str(o) )
