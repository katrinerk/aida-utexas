# This is for the first approach of Geolocation embedding augmentation. We extend the embeddings that exist in inderxers.p
import dill
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from gen_single_doc_graphs import Indexer
indexer_dir = "/home/cc/data/out_salads_70k_Indexed"

def getGeoInfo(target, df):
    target.replace("_"," ")
    if df[df["location"]==target].empty:
        return 0, 0
    else:  
        return df[df["location"]==target]["longitude_sc"].values, df[df["location"]==target]["latitude_sc"].values


if __name__ == "__main__":
    ere_indexer, stmt_indexer, ere_emb_mat, stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt = dill.load(open("/home/cc/data/out_salads_70k_Indexed_Geo/indexers_geo.p", 'rb'))
    new_ere_emb_mat = np.pad(ere_emb_mat, [(0,0),(0,2)], mode='constant')
    new_stmt_emb_mat = np.pad(stmt_emb_mat, [(0,0),(0,2)], mode='constant')

    # Load geo csv, normalize
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    df = pd.read_csv("/home/cc/aida-utexas/augment/crawled_geolocation_clean_full.csv")
    df = df[df['longitude'].notna()]
    df = df.drop_duplicates()
    x = df["longitude"].values.reshape(-1,1)
    df["longitude_sc"] = min_max_scaler.fit_transform(x)
    x = df["latitude"].values.reshape(-1,1)
    df["latitude_sc"] = min_max_scaler.fit_transform(x)
    breakpoint()
    count=0
    
    # For inderxers.p

    # Load indexer info
    
    print("Processing indexers.p...")
    for idx in ere_indexer.index_to_word.keys():
        target = ere_indexer.index_to_word[idx]
        if target[0].islower():
            continue
        lng, lat = getGeoInfo(ere_indexer.index_to_word[idx], df)
        print(ere_indexer.index_to_word[idx],lng, lat)
        if(lng!=0 and lat!=0):
        # if((lng & lat).all()):
            print(idx, ere_indexer.index_to_word[idx])
            count+=1
            ere_emb_mat[idx][-2] = lng
            ere_emb_mat[idx][-1] = lat
    
    dill.dump((ere_indexer, stmt_indexer, new_ere_emb_mat, new_stmt_emb_mat, num_word2vec_ere, num_word2vec_stmt), open(os.path.join(indexer_dir, "indexers_emb_augment.p"), "wb"))
