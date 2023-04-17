import csv
import geocoder
import collections
import time

start_idx = 9700
with open('geolocation_full.csv', mode='r') as csv_file:
    with open('crawled_geolocation_clean.csv', 'w') as csv_output_file:
        with open('crawled_geolocation_nodata.csv', 'w') as csv_output_file_2:
            csv_writer = csv.writer(csv_output_file)
            csv_writer.writerow(['location', 'longitude', 'latitude', 'continent', 'country', 'desc','wiki_url', 'res_addr', 'match'])
            csv_writer_2 = csv.writer(csv_output_file_2)
            csv_writer_2.writerow(['location', 'longitude', 'latitude', 'continent', 'country', 'desc','wiki_url', 'res_addr'])
            csv_reader = csv.DictReader(csv_file, fieldnames=['location'])
            line_count = 0
            for row in csv_reader:
                line_count+=1
                geoloc = row["location"]
                print(line_count, geoloc)
                if line_count < start_idx:
                    continue
                if(geoloc[0].islower()):
                    continue
                g = geocoder.geonames(geoloc, key='jiayingli', maxRows=1)
                # if g.json has return value but g.ok is false, then there must be error msg in json
                while(g.error):
                    print("Sleep 1 hour due to the api error:", g.error)
                    for i in range(3600,0,-1):
                        print(f"{i}", end="\r", flush=True)
                        time.sleep(1)
                    g = geocoder.geonames(geoloc, key='jiayingli', maxRows=1)

                for cand in g:
                    geoid = cand.geonames_id
                    # print(id, geoloc, geoid)
                    g = geocoder.geonames(geoid, method='details', key='jiayingli')
                    #geoInfo[id].append([geoloc, geoid, cand.description, g.continent, g.country, g.lat, g.lat, g.address, g.feature_class, g.class_description, g.wikipedia])
                    print([geoloc, geoid, cand.description, g.continent, g.country, g.lng, g.lat, g.address, g.feature_class, g.class_description, g.wikipedia])
                    if g.address == geoloc:
                        csv_writer.writerow([geoloc, g.lng, g.lat, g.continent, g.country, cand.description, g.wikipedia, g.address, True])
                    elif cand.description == "region":
                        csv_writer.writerow([geoloc, g.lng, g.lat, g.continent, g.country, cand.description, g.wikipedia, g.address, False])
                    else:
                        csv_writer_2.writerow([geoloc, g.lng, g.lat, g.continent, g.country, cand.description, g.wikipedia, g.address])                


