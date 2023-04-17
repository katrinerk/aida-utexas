import geocoder

geoloc = "Texas"
g = geocoder.geonames(geoloc, key='tycc0124', maxRows=3)

for cand in g:
    geoid = cand.geonames_id
    g = geocoder.geonames(geoid, method='details', key='tycc0124')
    #geoInfo[id].append([geoloc, geoid, cand.description, g.continent, g.country, g.lat, g.lat, g.address, g.feature_class, g.class_description, g.wikipedia])
    print([geoloc, geoid, cand.description, g.continent, g.country, g.lng, g.lat, g.address, g.feature_class, g.class_description, g.wikipedia])