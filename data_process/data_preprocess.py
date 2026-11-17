import os
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)


def preprocess(json_file_path, to_path):
    """Extract user/item pairs from a raw JSON review file."""
    fin = open(json_file_path, 'r')
    review_list = []
    i = 0
    for line in fin:
        try:
            if i % 10000 == 0:
                logger.info(i)
            d = eval(line, {"true": True, "false": False, "null": None})
            review_list.append([d['reviewerID'], d['parent_asin']])
        except Exception:
            continue
        i += 1
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id'])
    df.to_csv(to_path, index=False)


def filter_k_core(data, k_user=5, k_item=5, u_name='user_id', i_name='item_id', y_name='rating'):
    """
    Iterative k-core filtering until convergence.
    Removes users/items with fewer than k interactions repeatedly until stable.
    Single-pass (original behavior) was not guaranteed to converge.
    """
    while True:
        n_before = len(data)

        # Step 2: remove users with fewer than k_user interactions.
        user_counts = data.groupby(u_name)[y_name].count()
        data = data[data[u_name].isin(user_counts[user_counts >= k_user].index)]

        # Step 3: remove items with fewer than k_item interactions.
        item_counts = data.groupby(i_name)[y_name].count()
        data = data[data[i_name].isin(item_counts[item_counts >= k_item].index)]

        # Step 4: repeat Steps 2 and 3 until neither removes any rows.
        if len(data) == n_before:
            break
    return data


def find_common_users(df_source, df_target):
    """Return list of users present in both domain DataFrames."""
    return list(set(df_source['user_id'].unique()) & set(df_target['user_id'].unique()))


def filter_users_symmetric(inter_users, df_source, df_target, min_inter, u_name='user_id'):
    """
    Keep users with at least min_inter interactions in EACH domain.

    Symmetric filter for bidirectional CDR models: both domains are treated
    equally, so the dataset carries sufficient signal for transfer in either
    direction.  The resulting count is determined by the data distribution.
    """
    eligible = set(inter_users)
    src_counts = df_source[df_source[u_name].isin(eligible)].groupby(u_name).size()
    tgt_counts = df_target[df_target[u_name].isin(eligible)].groupby(u_name).size()
    eligible &= set(src_counts[src_counts >= min_inter].index)
    eligible &= set(tgt_counts[tgt_counts >= min_inter].index)
    logger.info(f"After symmetric min_inter={min_inter}: {len(eligible)} users")
    return list(eligible)


def id_map(source_path, user_to_path, item_to_path, to_path):
    """Assign contiguous integer IDs to users and items, save mappings and mapped data."""
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    df = pd.read_csv(source_path)
    uni_users = sorted(df['user_id'].unique().tolist())
    uni_items = sorted(df['item_id'].unique().tolist())

    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}

    u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['item_id', 'itemID'])
    df['userID'] = df['user_id'].map(u_id_map).astype(int)
    df['itemID'] = df['item_id'].map(i_id_map).astype(int)
    u_df.to_csv(user_to_path, index=False)
    i_df.to_csv(item_to_path, index=False)
    df.to_csv(to_path, index=False)


def build_dataset(df_source_raw, df_target_raw, source_out, target_out,
                  k_user=3, k_item=3,
                  min_inter=None,
                  u_name='user_id', i_name='item_id', y_name='rating'):
    """
    Full preprocessing pipeline for a cross-domain dataset pair.

        Steps:
            1. Extract users common to both raw domains and restrict both domains.
            2. Remove users with fewer than k_user interactions.
            3. Remove items with fewer than k_item interactions.
            4. Repeat Steps 2 and 3 independently in each domain until stable.
            5. Re-extract common users after filtering, restrict both domains, and save.

        Per the dataset procedure, k-core is not applied again after Step 5.

    Args:
        df_source_raw / df_target_raw : raw DataFrames with columns [user_id, item_id, rating, ...]
        source_out / target_out       : output CSV paths
        k_user / k_item               : k-core thresholds (default 3; lower = sparser but larger)
    min_inter                     : keep users with >= N interactions in EACH domain (symmetric)
    """
    logger.info(f"Source raw: {len(df_source_raw)}, Target raw: {len(df_target_raw)}")

    # Step 1: extract overlapping users before any k-core filtering.
    inter_users = find_common_users(df_source_raw, df_target_raw)
    logger.info(f"Initial common users: {len(inter_users)}")

    df_src = df_source_raw[df_source_raw[u_name].isin(inter_users)].copy()
    df_tgt = df_target_raw[df_target_raw[u_name].isin(inter_users)].copy()

    if min_inter is not None:
        inter_users = filter_users_symmetric(
            inter_users, df_src, df_tgt, min_inter=min_inter, u_name=u_name)
        df_src = df_src[df_src[u_name].isin(inter_users)]
        df_tgt = df_tgt[df_tgt[u_name].isin(inter_users)]

    # Steps 2-4: independently obtain the iterative k-core in each domain.
    df_src = filter_k_core(df_src, k_user=k_user, k_item=k_item,
                           u_name=u_name, i_name=i_name, y_name=y_name)
    df_tgt = filter_k_core(df_tgt, k_user=k_user, k_item=k_item,
                           u_name=u_name, i_name=i_name, y_name=y_name)
    logger.info(f"After {k_user}/{k_item}-core — Source: {len(df_src)}, Target: {len(df_tgt)}")

    # Step 5: re-extract overlapping users. Do not run k-core after this step.
    inter_users = find_common_users(df_src, df_tgt)
    logger.info(f"Final common users: {len(inter_users)}")
    df_src = df_src[df_src[u_name].isin(inter_users)]
    df_tgt = df_tgt[df_tgt[u_name].isin(inter_users)]

    logger.info(
        f"Final — Source: {len(df_src)} interactions "
        f"({df_src[u_name].nunique()} users, {df_src[i_name].nunique()} items)"
    )
    logger.info(
        f"Final — Target: {len(df_tgt)} interactions "
        f"({df_tgt[u_name].nunique()} users, {df_tgt[i_name].nunique()} items)"
    )

    os.makedirs(os.path.dirname(source_out), exist_ok=True)
    os.makedirs(os.path.dirname(target_out), exist_ok=True)
    df_src.to_csv(source_out, index=False)
    df_tgt.to_csv(target_out, index=False)
    return df_src, df_tgt


DATASET_CONFIG = {
    'cloth_sport': {
        'source_file': 'Clothing_Shoes_and_Jewelry.csv.gz',
        'target_file': 'Sports_and_Outdoors.csv.gz',
        'source_name': 'cloth',
        'target_name': 'sport',
    },
    'elec_cloth': {
        'source_file': 'Electronics.csv.gz',
        'target_file': 'Clothing_Shoes_and_Jewelry.csv.gz',
        'source_name': 'elec',
        'target_name': 'cloth',
    },
    'phone_sport': {
        'source_file': 'Cell_Phones_and_Accessories.csv.gz',
        'target_file': 'Sports_and_Outdoors.csv.gz',
        'source_name': 'phone',
        'target_name': 'sport',
    },
    'sport_clothing': {
        'source_file': 'Clothing_Shoes_and_Jewelry.csv.gz',
        'target_file': 'Sports_and_Outdoors.csv.gz',
        'source_name': 'clothing',
        'target_name': 'sport',
    },
    'movie_music': {
        'source_file': 'Movies_and_TV.csv.gz',
        'target_file': 'CDs_and_Vinyl.csv.gz',
        'source_name': 'movie',
        'target_name': 'music',
    },
    'toy_sport': {
        'source_file': 'Toys_and_Games.csv.gz',
        'target_file': 'Sports_and_Outdoors.csv.gz',
        'source_name': 'toy',
        'target_name': 'sport',
    },
}

DOMAIN_FILES = {
    'movie': 'Movies_and_TV.csv.gz',
    'music': 'CDs_and_Vinyl.csv.gz',
    'book': 'Books.csv.gz',
    'phone': 'Cell_Phones_and_Accessories.csv.gz',
    'cloth': 'Clothing_Shoes_and_Jewelry.csv.gz',
    'elec': 'Electronics.csv.gz',
    'sport': 'Sports_and_Outdoors.csv.gz',
    'toy': 'Toys_and_Games.csv.gz',
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toy_sport',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--source', choices=list(DOMAIN_FILES),
                        help='Source domain for an arbitrary pair (use with --target)')
    parser.add_argument('--target', choices=list(DOMAIN_FILES),
                        help='Target domain for an arbitrary pair (use with --source)')
    parser.add_argument('--k', type=int, default=None,
                        help='Min interactions per user/item for k-core filter (sets both k_user and k_item)')
    parser.add_argument('--k_user', type=int, default=10,
                        help='Min interactions per user for k-core filter (default 10)')
    parser.add_argument('--k_item', type=int, default=10,
                        help='Min interactions per item for k-core filter (default 10)')
    parser.add_argument('--min_inter', type=int, default=None,
                        help='Keep users with >= N interactions in EACH domain (symmetric, for bidirectional CDR)')
    args = parser.parse_args()

    if args.k is not None:
        args.k_user = args.k
        args.k_item = args.k

    if (args.source is None) != (args.target is None):
        parser.error('--source and --target must be specified together')
    if args.source is not None:
        if args.source == args.target:
            parser.error('--source and --target must be different')
        cfg = {
            'source_file': DOMAIN_FILES[args.source],
            'target_file': DOMAIN_FILES[args.target],
            'source_name': args.source,
            'target_name': args.target,
        }
        pair_name = f'{args.source}_{args.target}'
    else:
        cfg = DATASET_CONFIG[args.dataset]
        pair_name = args.dataset

    datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    base = os.path.join(datasets_dir, pair_name)
    src_name = cfg['source_name']
    tgt_name = cfg['target_name']
    columns = ['user_id', 'item_id', 'rating', 'timestamp']

    df_source_raw = pd.read_csv(os.path.join(datasets_dir, cfg['source_file']))
    df_source_raw.columns = columns
    df_target_raw = pd.read_csv(os.path.join(datasets_dir, cfg['target_file']))
    df_target_raw.columns = columns

    build_dataset(
        df_source_raw, df_target_raw,
        source_out=f'{base}/{src_name}/data_{src_name}.csv',
        target_out=f'{base}/{tgt_name}/data_{tgt_name}.csv',
        k_user=args.k_user,
        k_item=args.k_item,
        min_inter=args.min_inter,
    )

    id_map(
        source_path=f'{base}/{src_name}/data_{src_name}.csv',
        user_to_path=f'{base}/{src_name}/user_id_map.csv',
        item_to_path=f'{base}/{src_name}/item_id_map.csv',
        to_path=f'{base}/{src_name}/{src_name}_inter.csv',
    )
    id_map(
        source_path=f'{base}/{tgt_name}/data_{tgt_name}.csv',
        user_to_path=f'{base}/{tgt_name}/user_id_map.csv',
        item_to_path=f'{base}/{tgt_name}/item_id_map.csv',
        to_path=f'{base}/{tgt_name}/{tgt_name}_inter.csv',
    )
