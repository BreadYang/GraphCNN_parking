#################
# Convert Road network to a directed graph
# Generating node correlation matrix for the graph
# Document: https://arxiv.org/abs/1901.06758
# Last update: 2018/01/01 
#################

import math
import csv
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import collections
from geojson import Point, Feature, FeatureCollection

googlemap_key = ""
Features = []
term_2_link = {}
links = {}
oldlink = ''

ofile_d = open("link_latlon.csv", 'wb')
writer_d = csv.writer(ofile_d, delimiter=',', lineterminator='\n', dialect='excel')
writer_d.writerow(['Link ID', 'Location', "Latitude", "Longitude", 'Terminal_list'])
Link_geo = {}
with open("downtown_links.csv") as infile:
    words = infile.readline().replace("\n", "").split(',')
    link_lat = []
    link_lon = []
    TermList = []                
    for line in infile:
        words = line.replace("\n", "").split(',')
        Terminal = str(words[0])
        link = str(words[5])
        loc = str(words[2])
        if link != oldlink and oldlink != '':
            lat = np.average(link_lat)
            lon = np.average(link_lon)
            Terms = '_'.join(TermList)
            writer_d.writerow([oldlink, oldloc, str(lat), str(lon), Terms])
            mypoint = Point((lon, lat))
            myfeature = Feature(geometry=mypoint, properties={"Link_ID": oldlink, "Terminals": Terms,
                                      "Location": oldloc})
            Features.append(myfeature)
            Link_geo[oldlink] = (lat, lon)
               ###
            link_lat = []
            link_lon = []
            TermList = []  
        link_lat.append(np.float(words[3]))
        link_lon.append(np.float(words[4]))
        TermList.append(Terminal)
        oldlink = link
        oldloc = loc
        
lat = np.average(link_lat)
lon = np.average(link_lon)
Terms = '_'.join(TermList)
writer_d.writerow([oldlink, oldloc, str(lat), str(lon), Terms])
mypoint = Point((lon, lat))
myfeature = Feature(geometry=mypoint, properties={"Link_ID": oldlink, "Terminals": Terms,
"Location": oldloc})
Features.append(myfeature)
Link_geo[oldlink] = (lat, lon)

output = FeatureCollection(Features)
ofile_geo = open("link_latlon.geojson", 'wb')
ofile_geo.write(str(output))
ofile_geo.close()
ofile_d.close()

import googlemaps
import json
import numpy as np
import time 

gmaps = googlemaps.Client(key=googlemap_key)
station_list = Link_geo.keys()
station_list.sort()
n = len(station_list)

ofile_d = open("Dist_matrix.csv", 'wb')
writer_d = csv.writer(ofile_d, delimiter=',', lineterminator='\n', dialect='excel')
writer_d.writerow(["origin"]+ station_list)
origin = station_list[0]
dest = station_list[1]
js = gmaps.distance_matrix(origins = Link_geo[origin], destinations = Link_geo[dest], mode = 'driving')
for origin in station_list:
    dist_array = []
    for dest in station_list:
        js = gmaps.distance_matrix(origins = Link_geo[origin], destinations = Link_geo[dest], mode = 'driving')
        distance = (js['rows'][0]['elements'][0]['duration']['value'])
        dist_array.append(distance)
    print dist_array
    time.sleep(61)
    writer_d.writerow([origin]+dist_array)