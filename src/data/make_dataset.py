# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import multiprocessing as mp
import os

import numpy as np
import glob
import pandas as pd
import tqdm
from dotenv import find_dotenv, load_dotenv
from inflection import underscore

from constants import *


def _gather_parsed_files() -> list:
    """Gather all parsed json filenames

    Returns:
        list: list of all json demofiles
    """
    parsed_json_files = glob.glob('./data/interim/*.json')
    return parsed_json_files


def _build_df_from_parsed(parsed_file: str) -> pd.DataFrame:
    """Generate a clean pandas Dataframe from a structured parsed demofile

    Args:
        parsed_file (dict): the parsed json demofile as a structured dictionary

    Returns:
        pd.DataFrame: a pandas dataframe
    """
    try:
        with open(parsed_file, 'r') as fp:
            data = json.load(fp)

        filtered_game_rounds = [{key: round[key]
                                 for key in round_var_filter} for round in data['gameRounds']]

        df = pd.DataFrame(filtered_game_rounds)

        for column in meta_columns:
            df[column] = data[column]

        def __hash(row):
            concat_str = ''.join([str(x) for x in np.array(row)])
            return hashlib.sha3_256(concat_str.encode('utf-8')).hexdigest()[:32]

        df = df[column_order]
        df['_data_id'] = df.apply(lambda x: __hash(x), axis=1)
        df['_match_id'] = df[['matchID', 'clientName', 'mapName']].apply(
            lambda x: __hash(x), axis=1)
        df.reset_index(inplace=True, drop=True)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f'{parsed_file} failed to parse ... reason: \n {e}')
        df = pd.DataFrame()

    return df


def _build_df_from_list(filenames: list) -> pd.DataFrame:
    """Generate a pandas dataframe from list of filenames

    Args:
        filenames (list): list of filenames

    Returns:
        pd.DataFrame: a pandas dataframe
    """
    cpu_count = os.cpu_count()
    cpu_count -= 1
    # Ensure at least 1 core
    if cpu_count <= 1:
        cpu_count = 1
    
    with mp.Pool(cpu_count) as pool:
        # data = pool.map(_build_df_from_parsed, filenames)
        # Display Progress Bar
        data = list()
        for _df in tqdm.tqdm(pool.imap_unordered(_build_df_from_parsed, filenames), total=len(filenames), desc='👷‍♀️ Building df', unit=' files'):
            data.append(_df)
    
    df = pd.concat(data, ignore_index=True)
    return df


def _clean_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the given dataframe for further processing and analysis

    Args:
        df (pd.DataFrame): A raw dataframe of the team data
    
    :rtype: pd.DataFrame
    """
    df.rename(columns={"matchID": "match_name"}, inplace=True)
    columns = df.columns
    snake_case_columns = [underscore(c) for c in columns]
    df.columns = snake_case_columns

    buy_types = ['Full Eco', 'Full Buy', 'Half Buy', 'Eco']

    # Ensure there are now draws and ties (not really possible but parsing errors might occur)
    df = df.loc[(df['t_score'] != df['end_t_score']) |
                (df['ct_score'] != df['end_ct_score'])]

    # Check that the rounds aren't warmup rounds
    df = df[df['is_warmup'] == False]

    # Ensure only valid Buy Types
    df = df[df['t_buy_type'].isin(buy_types)]
    df = df[df['ct_buy_type'].isin(buy_types)]

    # Define Scoring Variable for each Round for further analysis
    df['t_round_win'] = df['end_t_score'] - df['t_score']
    df['ct_round_win'] = df['end_ct_score'] - df['ct_score']

    # Ensure scoring variable is in valid range
    df = df[df['t_round_win'].isin([0, 1])]
    df = df[df['ct_round_win'].isin([0, 1])]
    print(f'Data looks like this: \n{df}')
    return df


def _save_dataset(df: pd.DataFrame) -> None:
    """Stores the dataframe as a feather file.

    Args:
        df (pd.DataFrame): The cleaned data as a dataframe
    """
    df.reset_index(inplace=True)
    df.to_feather('./data/processed/team_score_and_buy__dataset.feather')


def main():
    """ Runs data processing scripts to turn preparsed data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('MAKING FINAL DATASET FROM THE RAW DATA')

    # Step 1: Gather all parsed Files
    logger.info('Step 1/4: Gathering all parsed Files')
    parsed_files = _gather_parsed_files()

    # Step 2: Build df from files
    logger.info('Step 2/4: build initial df')
    raw_df = _build_df_from_list(parsed_files)

    # Step 3: clean and prepare the df
    logger.info('Step 3/4: clean and prepare the df')
    cleaned_df = _clean_and_prepare_df(raw_df)

    # Step 4: store the final dataset as a feather file
    logger.info('Step 4/4: store the final dataset as a feather file')
    _save_dataset(cleaned_df)

    logger.info(' ✅ Final Dataset created!')
    logger.info(
        f'💾 File saved to: ./data/processed/team_score_and_buy__dataset.feather')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
