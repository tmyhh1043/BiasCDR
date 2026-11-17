import pandas as pd
import gzip
from sentence_transformers	import SentenceTransformer
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
import torch

def select_data(df,meta_path,to_path):
    items = set(df['item_id'].unique())
    with gzip.open(meta_path, 'rb') as g:
        for line in tqdm(g, desc="Processing lines", ascii=True):
            d = eval(line, {"true":True,"false":False,"null":None})
            if 'parent_asin' in d and d['parent_asin'] in items:
                try:
                    item_id = d['parent_asin']
                except:
                    item_id = ' '
                try:
                    im_url = d['images']
                except:
                    im_url = '[]'
                try:
                    description = d['description']
                except:
                    description = ' '
                try:
                    categories = ','.join(d['categories'])
                except:
                    categories = ' '
                try:
                    title = d['title']
                except:
                    title = ' '
                each_line = {
                    'item_id':[item_id],
                    'im_url':[im_url],
                    'description':[description],
                    'categories':[categories],
                    'title':[title]
                }
                each_line_df = pd.DataFrame(each_line)
                each_line_df.to_csv(to_path,mode='a',header=False,index=False, encoding='utf-8', errors='replace')

        
def process_text_data(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['item_id'], inplace=True)
    orig_item_ids  = df['item_id'].unique().tolist()
    # Read with dtype=str to handle mixed types in columns 1 and 3
    df_text = pd.read_csv(source_path_text)
    df_text.columns = ['item_id', 'im_url', 'description', 'categories', 'title']
    logger.info('process null value')
    df_text.sort_values(by=['item_id'], inplace=True)
    # Replace NaN and empty strings with appropriate empty values
    df_text['im_url'] = df_text['im_url'].fillna("[]")
    df_text['description'] = df_text['description'].fillna(" ")
    df_text['title'] = df_text['title'].fillna(" ")
    df_text['categories'] = df_text['categories'].fillna(" ")
    sentences = []
    lack_index = []
    for each_item in orig_item_ids:
        row = df_text[df_text['item_id']==each_item]
        if len(row) == 0:
            print(each_item)
            lack_index.append(orig_item_ids.index(each_item))
        else:
            sen = row['title'].iloc[0] + ' ' + row['categories'].iloc[0] + ' ' + row['description'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    logger.info('start transform')
    # Check if CUDA is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    sentence_embeddings = model.encode(sentences)
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done!')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='phone_sport',
                        choices=['phone_sport', 'sport_clothing', 'toy_sport', 'movie_music',
                                 'elec_cloth', 'cloth_sport'])
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == 'cloth_sport':
        df = pd.read_csv('../datasets/cloth_sport/cloth/cloth_inter.csv')
        select_data(df, meta_path='../datasets/meta_Clothing_Shoes_and_Jewelry.jsonl.gz',
                    to_path='../datasets/cloth_sport/cloth/meta_cloth_data.csv')
        process_text_data(source_path='../datasets/cloth_sport/cloth/cloth_inter.csv',
                          source_path_text='../datasets/cloth_sport/cloth/meta_cloth_data.csv',
                          to_path='../datasets/cloth_sport/cloth/cloth_text_feat.npy')

        df = pd.read_csv('../datasets/cloth_sport/sport/sport_inter.csv')
        select_data(df, meta_path='../datasets/meta_Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/cloth_sport/sport/meta_sport_data.csv')
        process_text_data(source_path='../datasets/cloth_sport/sport/sport_inter.csv',
                          source_path_text='../datasets/cloth_sport/sport/meta_sport_data.csv',
                          to_path='../datasets/cloth_sport/sport/sport_text_feat.npy')

    elif dataset == 'elec_cloth':
        df = pd.read_csv('../datasets/elec_cloth/elec/elec_inter.csv')
        select_data(df, meta_path='../datasets/meta_Electronics.jsonl.gz',
                    to_path='../datasets/elec_cloth/elec/meta_elec_data.csv')
        process_text_data(source_path='../datasets/elec_cloth/elec/elec_inter.csv',
                          source_path_text='../datasets/elec_cloth/elec/meta_elec_data.csv',
                          to_path='../datasets/elec_cloth/elec/elec_text_feat.npy')

        df = pd.read_csv('../datasets/elec_cloth/cloth/cloth_inter.csv')
        select_data(df, meta_path='../datasets/meta_Clothing_Shoes_and_Jewelry.jsonl.gz',
                    to_path='../datasets/elec_cloth/cloth/meta_cloth_data.csv')
        process_text_data(source_path='../datasets/elec_cloth/cloth/cloth_inter.csv',
                          source_path_text='../datasets/elec_cloth/cloth/meta_cloth_data.csv',
                          to_path='../datasets/elec_cloth/cloth/cloth_text_feat.npy')

    elif dataset == 'phone_sport':
        df = pd.read_csv('../datasets/phone_sport/phone/phone_inter.csv')
        select_data(df,meta_path='../datasets/meta_Cell_Phones_and_Accessories.jsonl.gz',
                    to_path='../datasets/phone_sport/phone/meta_phone_data.csv')
        process_text_data(source_path='../datasets/phone_sport/phone/phone_inter.csv',
                        source_path_text='../datasets/phone_sport/phone/meta_phone_data.csv',
                        to_path='../datasets/phone_sport/phone/phone_text_feat.npy')

        df = pd.read_csv('../datasets/phone_sport/sport/sport_inter.csv')
        select_data(df,meta_path='../datasets/meta_Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/phone_sport/sport/meta_sport_data.csv')
        process_text_data(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                        source_path_text='../datasets/phone_sport/sport/meta_sport_data.csv',
                        to_path='../datasets/phone_sport/sport/sport_text_feat.npy')

    elif dataset == 'sport_clothing':
        df = pd.read_csv('../datasets/sport_clothing/sport/sport_inter.csv')
        select_data(df,meta_path='../datasets/meta_Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/sport_clothing/sport/meta_sport_data.csv')
        process_text_data(source_path='../datasets/sport_clothing/sport/sport_inter.csv',
                          source_path_text='../datasets/sport_clothing/sport/meta_sport_data.csv',
                          to_path='../datasets/sport_clothing/sport/sport_text_feat.npy')

        df = pd.read_csv('../datasets/sport_clothing/clothing/clothing_inter.csv')
        select_data(df,meta_path='../datasets/meta_Clothing_Shoes_and_Jewelry.jsonl.gz',
                    to_path='../datasets/sport_clothing/clothing/meta_clothing_data.csv')
        process_text_data(source_path='../datasets/sport_clothing/clothing/clothing_inter.csv',
                          source_path_text='../datasets/sport_clothing/clothing/meta_clothing_data.csv',
                          to_path='../datasets/sport_clothing/clothing/clothing_text_feat.npy')

    elif dataset == 'toy_sport':
        df = pd.read_csv('../datasets/toy_sport/toy/toy_inter.csv')
        select_data(df, meta_path='../datasets/meta_Toys_and_Games.jsonl.gz',
                    to_path='../datasets/toy_sport/toy/meta_toy_data.csv')
        process_text_data(source_path='../datasets/toy_sport/toy/toy_inter.csv',
                          source_path_text='../datasets/toy_sport/toy/meta_toy_data.csv',
                          to_path='../datasets/toy_sport/toy/toy_text_feat.npy')

        df = pd.read_csv('../datasets/toy_sport/sport/sport_inter.csv')
        select_data(df, meta_path='../datasets/meta_Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/toy_sport/sport/meta_sport_data.csv')
        process_text_data(source_path='../datasets/toy_sport/sport/sport_inter.csv',
                          source_path_text='../datasets/toy_sport/sport/meta_sport_data.csv',
                          to_path='../datasets/toy_sport/sport/sport_text_feat.npy')

    elif dataset == 'movie_music':
        df = pd.read_csv('../datasets/movie_music/movie/movie_inter.csv')
        select_data(df, meta_path='../datasets/meta_Movies_and_TV.jsonl.gz',
                    to_path='../datasets/movie_music/movie/meta_movie_data.csv')
        process_text_data(source_path='../datasets/movie_music/movie/movie_inter.csv',
                          source_path_text='../datasets/movie_music/movie/meta_movie_data.csv',
                          to_path='../datasets/movie_music/movie/movie_text_feat.npy')

        df = pd.read_csv('../datasets/movie_music/music/music_inter.csv')
        select_data(df, meta_path='../datasets/meta_CDs_and_Vinyl.jsonl.gz',
                    to_path='../datasets/movie_music/music/meta_music_data.csv')
        process_text_data(source_path='../datasets/movie_music/music/music_inter.csv',
                          source_path_text='../datasets/movie_music/music/meta_music_data.csv',
                          to_path='../datasets/movie_music/music/music_text_feat.npy')
