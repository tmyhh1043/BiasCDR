import array

import pandas as pd
import numpy as np
from loguru import logger


def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin_bytes = f.read(10)
        if len(asin_bytes) < 10:
            break
        asin = asin_bytes.decode('UTF-8').strip()
        feature_bytes = f.read(4096 * 4)
        if len(feature_bytes) < 4096 * 4:
            print("Incomplete record encountered. Ending reading.")
            break
        
        a = array.array('f')
        a.frombytes(feature_bytes)
        yield asin, a.tolist()
def process_visual_feat(source_path,source_image_path,to_path):
  logger.info('read csv')
  df = pd.read_csv(source_path)
  df.sort_values(by=['item_id'], inplace=True)
  item2id = df['item_id'].unique().tolist()
  img_data = readImageFeatures(source_image_path)
  feats = {}
  avg = []
  logger.info('start image data')
  for d in img_data:
    if d[0] in item2id:
      feats[d[0]] = d[1]
      avg.append(d[1])
  avg = np.array(avg).mean(0).tolist()

  ret = []
  non_no = []
  logger.info('start filter')
  for i in item2id:
    if i in feats:
      ret.append(feats[i])
    else:
      non_no.append(i)
      ret.append(avg)

  logger.info('# of items not in processed image features:', len(non_no))
  assert len(ret) == len(item2id)
  np.save(to_path, np.array(ret))
  logger.info('complete')

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='phone_sport',
                      choices=['phone_sport', 'sport_clothing', 'toy_sport', 'movie_music',
                               'elec_cloth', 'cloth_sport'])
  args = parser.parse_args()
  dataset = args.dataset

  if dataset == 'cloth_sport':
    process_visual_feat(source_path='../datasets/cloth_sport/cloth/cloth_inter.csv',
                        source_image_path='../datasets/cloth_sport/cloth/image_features_Clothing_Shoes_and_Jewelry.b',
                        to_path='../datasets/cloth_sport/cloth/cloth_visual_feat.npy')
    process_visual_feat(source_path='../datasets/cloth_sport/sport/sport_inter.csv',
                        source_image_path='../datasets/cloth_sport/sport/image_features_Sports_and_Outdoors.b',
                        to_path='../datasets/cloth_sport/sport/sport_visual_feat.npy')

  elif dataset == 'elec_cloth':
    process_visual_feat(source_path='../datasets/elec_cloth/elec/elec_inter.csv',
                        source_image_path='../datasets/elec_cloth/elec/image_features_Electronics.b',
                        to_path='../datasets/elec_cloth/elec/elec_visual_feat.npy')
    process_visual_feat(source_path='../datasets/elec_cloth/cloth/cloth_inter.csv',
                        source_image_path='../datasets/elec_cloth/cloth/image_features_Clothing_Shoes_and_Jewelry.b',
                        to_path='../datasets/elec_cloth/cloth/cloth_visual_feat.npy')

  elif dataset == 'phone_sport':
    # process_visual_feat(source_path='../datasets/phone_sport/phone/phone_inter.csv',
    #                     source_image_path='../datasets/phone_sport/phone/image_features_Cell_Phones_and_Accessories.b',
    #                     to_path='../datasets/phone_sport/phone/phone_visual_feat.npy')

    process_visual_feat(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                        source_image_path='../datasets/phone_sport/sport/image_features_Sports_and_Outdoors.b',
                        to_path='../datasets/phone_sport/sport/sport_visual_feat.npy')
    
  elif dataset == 'sport_clothing':
    process_visual_feat(source_path='../datasets/sport_clothing/sport/sport_inter.csv',
                        source_image_path='../datasets/sport_clothing/sport/image_features_Sports_and_Outdoors.b',
                        to_path='../datasets/sport_clothing/sport/sport_visual_feat.npy')
    process_visual_feat(source_path='../datasets/sport_clothing/clothing/clothing_inter.csv',
                        source_image_path='../datasets/sport_clothing/clothing/image_features_Clothing.b',
                        to_path='../datasets/sport_clothing/clothing/clothing_visual_feat.npy')

  elif dataset == 'toy_sport':
    process_visual_feat(source_path='../datasets/toy_sport/toy/toy_inter.csv',
                        source_image_path='../datasets/toy_sport/toy/image_features_Toys_and_Games.b',
                        to_path='../datasets/toy_sport/toy/toy_visual_feat.npy')
    process_visual_feat(source_path='../datasets/toy_sport/sport/sport_inter.csv',
                        source_image_path='../datasets/toy_sport/sport/image_features_Sports_and_Outdoors.b',
                        to_path='../datasets/toy_sport/sport/sport_visual_feat.npy')

  elif dataset == 'movie_music':
    process_visual_feat(source_path='../datasets/movie_music/movie/movie_inter.csv',
                        source_image_path='../datasets/movie_music/movie/image_features_Movies_and_TV.b',
                        to_path='../datasets/movie_music/movie/movie_visual_feat.npy')
    process_visual_feat(source_path='../datasets/movie_music/music/music_inter.csv',
                        source_image_path='../datasets/movie_music/music/image_features_CDs_and_Vinyl.b',
                        to_path='../datasets/movie_music/music/music_visual_feat.npy')
