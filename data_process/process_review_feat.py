import gzip
import pandas as pd
from sentence_transformers	import SentenceTransformer
import numpy as np
from loguru import logger

def select_reviews(source_path,to_path,items,users):
    g = gzip.open(source_path, 'r')
    review_list = []
    i = 0
    for line in g:
        d = eval(line, {"true": True, "false": False, "null": None})
        if (d['parent_asin'] in items) and (d['user_id'] in users):
            if i % 10000 == 0:
                logger.info(i)
            i+=1
            review_list.append([d['user_id'], d['parent_asin'],d['text']])
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id','review_text'])  
    df.to_csv(to_path, index=False, encoding='utf-8', errors='replace')

def concat_reviews(source_path,to_path):
    df = pd.read_csv(source_path)
    df['review_text'].fillna(' ', inplace=True)
    df_lists = []
    logger.info('start group by')
    for x in df.groupby(by='user_id'):
        each_df = pd.DataFrame({
            'user_id': [x[0]],
            'review_texts': [';'.join(x[1]['review_text'])]
        })
        df_lists.append(each_df)

    df = pd.concat(df_lists, axis=0)
    logger.info('start store')
    df.to_csv(to_path, index=False, encoding='utf-8', errors='replace')

def generate_review_emb(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['user_id'], inplace=True)
    orig_user_ids  = df['user_id'].unique().tolist()
    df_text = pd.read_csv(source_path_text)
    logger.info('process null value')
    df_text.sort_values(by=['user_id'], inplace=True)
    sentences = []
    lack_index = []
    for each_user in orig_user_ids:
        row = df_text[df_text['user_id']==each_user]
        if len(row) == 0:
            print(each_user)
            lack_index.append(orig_user_ids.index(each_user))
        else:
            sen = row['review_texts'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='phone_sport',
                        choices=['phone_sport', 'sport_clothing', 'toy_sport', 'movie_music',
                                 'elec_cloth', 'cloth_sport'])
    args = parser.parse_args()
    datasets = args.dataset

    if datasets == 'cloth_sport':
        df = pd.read_csv('../datasets/cloth_sport/cloth/cloth_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Clothing_Shoes_and_Jewelry.jsonl.gz',
                       to_path='../datasets/cloth_sport/cloth/cloth_reviews_orig.csv',
                       items=items, users=users)
        concat_reviews(source_path='../datasets/cloth_sport/cloth/cloth_reviews_orig.csv',
                       to_path='../datasets/cloth_sport/cloth/cloth_reviews.csv')
        generate_review_emb(source_path='../datasets/cloth_sport/cloth/cloth_inter.csv',
                            source_path_text='../datasets/cloth_sport/cloth/cloth_reviews.csv',
                            to_path='../datasets/cloth_sport/cloth/cloth_review_feat.npy')

        df = pd.read_csv('../datasets/cloth_sport/sport/sport_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Sports_and_Outdoors.jsonl.gz',
                       to_path='../datasets/cloth_sport/sport/sport_reviews_orig.csv',
                       items=items, users=users)
        concat_reviews(source_path='../datasets/cloth_sport/sport/sport_reviews_orig.csv',
                       to_path='../datasets/cloth_sport/sport/sport_reviews.csv')
        generate_review_emb(source_path='../datasets/cloth_sport/sport/sport_inter.csv',
                            source_path_text='../datasets/cloth_sport/sport/sport_reviews.csv',
                            to_path='../datasets/cloth_sport/sport/sport_review_feat.npy')

    elif datasets == 'elec_cloth':
        df = pd.read_csv('../datasets/elec_cloth/elec/elec_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Electronics.jsonl.gz',
                       to_path='../datasets/elec_cloth/elec/elec_reviews_orig.csv',
                       items=items, users=users)
        concat_reviews(source_path='../datasets/elec_cloth/elec/elec_reviews_orig.csv',
                       to_path='../datasets/elec_cloth/elec/elec_reviews.csv')
        generate_review_emb(source_path='../datasets/elec_cloth/elec/elec_inter.csv',
                            source_path_text='../datasets/elec_cloth/elec/elec_reviews.csv',
                            to_path='../datasets/elec_cloth/elec/elec_review_feat.npy')

        df = pd.read_csv('../datasets/elec_cloth/cloth/cloth_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Clothing_Shoes_and_Jewelry.jsonl.gz',
                       to_path='../datasets/elec_cloth/cloth/cloth_reviews_orig.csv',
                       items=items, users=users)
        concat_reviews(source_path='../datasets/elec_cloth/cloth/cloth_reviews_orig.csv',
                       to_path='../datasets/elec_cloth/cloth/cloth_reviews.csv')
        generate_review_emb(source_path='../datasets/elec_cloth/cloth/cloth_inter.csv',
                            source_path_text='../datasets/elec_cloth/cloth/cloth_reviews.csv',
                            to_path='../datasets/elec_cloth/cloth/cloth_review_feat.npy')

    elif datasets == 'phone_sport':
        df = pd.read_csv('../datasets/phone_sport/phone/phone_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Cell_Phones_and_Accessories.jsonl.gz',
                    to_path='../datasets/phone_sport/phone/phone_reviews_orig.csv', items=items, users=users)
        # # second step
        concat_reviews(source_path='../datasets/phone_sport/phone/phone_reviews_orig.csv',
                    to_path='../datasets/phone_sport/phone/phone_reviews.csv')
        generate_review_emb(source_path='../datasets/phone_sport/phone/phone_inter.csv',
                            source_path_text='../datasets/phone_sport/phone/phone_reviews.csv',
                            to_path='../datasets/phone_sport/phone/phone_review_feat.npy')

        df = pd.read_csv('../datasets/phone_sport/sport/sport_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/phone_sport/sport/sport_reviews_orig.csv', items=items, users=users)
        # # second step
        concat_reviews(source_path='../datasets/phone_sport/sport/sport_reviews_orig.csv',
                    to_path='../datasets/phone_sport/sport/sport_reviews.csv')
        generate_review_emb(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                            source_path_text='../datasets/phone_sport/sport/sport_reviews.csv',
                            to_path='../datasets/phone_sport/sport/sport_review_feat.npy')

    elif datasets == 'sport_clothing':
        df = pd.read_csv('../datasets/sport_clothing/sport/sport_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/sport_clothing/sport/sport_reviews_orig.csv', items=items, users=users)
        # # second step
        concat_reviews(source_path='../datasets/sport_clothing/sport/sport_reviews_orig.csv',
                    to_path='../datasets/sport_clothing/sport/sport_reviews.csv')
        generate_review_emb(source_path='../datasets/sport_clothing/sport/sport_inter.csv',
                            source_path_text='../datasets/sport_clothing/sport/sport_reviews.csv',
                            to_path='../datasets/sport_clothing/sport/sport_review_feat.npy')

        df = pd.read_csv('../datasets/sport_clothing/clothing/clothing_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Clothing_Shoes_and_Jewelry.jsonl.gz',
                    to_path='../datasets/sport_clothing/clothing/clothing_reviews_orig.csv', items=items, users=users)
        # # second step
        concat_reviews(source_path='../datasets/sport_clothing/clothing/clothing_reviews_orig.csv',
                    to_path='../datasets/sport_clothing/clothing/clothing_reviews.csv')
        generate_review_emb(source_path='../datasets/sport_clothing/clothing/clothing_inter.csv',
                            source_path_text='../datasets/sport_clothing/clothing/clothing_reviews.csv',
                            to_path='../datasets/sport_clothing/clothing/clothing_review_feat.npy')

    elif datasets == 'toy_sport':
        df = pd.read_csv('../datasets/toy_sport/toy/toy_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Toys_and_Games.jsonl.gz',
                       to_path='../datasets/toy_sport/toy/toy_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/toy_sport/toy/toy_reviews_orig.csv',
                       to_path='../datasets/toy_sport/toy/toy_reviews.csv')
        generate_review_emb(source_path='../datasets/toy_sport/toy/toy_inter.csv',
                            source_path_text='../datasets/toy_sport/toy/toy_reviews.csv',
                            to_path='../datasets/toy_sport/toy/toy_review_feat.npy')

        df = pd.read_csv('../datasets/toy_sport/sport/sport_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Sports_and_Outdoors.jsonl.gz',
                       to_path='../datasets/toy_sport/sport/sport_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/toy_sport/sport/sport_reviews_orig.csv',
                       to_path='../datasets/toy_sport/sport/sport_reviews.csv')
        generate_review_emb(source_path='../datasets/toy_sport/sport/sport_inter.csv',
                            source_path_text='../datasets/toy_sport/sport/sport_reviews.csv',
                            to_path='../datasets/toy_sport/sport/sport_review_feat.npy')

    elif datasets == 'movie_music':
        df = pd.read_csv('../datasets/movie_music/movie/movie_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Movies_and_TV.jsonl.gz',
                       to_path='../datasets/movie_music/movie/movie_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/movie_music/movie/movie_reviews_orig.csv',
                       to_path='../datasets/movie_music/movie/movie_reviews.csv')
        generate_review_emb(source_path='../datasets/movie_music/movie/movie_inter.csv',
                            source_path_text='../datasets/movie_music/movie/movie_reviews.csv',
                            to_path='../datasets/movie_music/movie/movie_review_feat.npy')

        df = pd.read_csv('../datasets/movie_music/music/music_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/CDs_and_Vinyl.jsonl.gz',
                       to_path='../datasets/movie_music/music/music_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/movie_music/music/music_reviews_orig.csv',
                       to_path='../datasets/movie_music/music/music_reviews.csv')
        generate_review_emb(source_path='../datasets/movie_music/music/music_inter.csv',
                            source_path_text='../datasets/movie_music/music/music_reviews.csv',
                            to_path='../datasets/movie_music/music/music_review_feat.npy')




