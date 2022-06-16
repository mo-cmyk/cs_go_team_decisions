# -*- coding: utf-8 -*-
import glob
import logging
import multiprocessing as mp
import os
import time

from awpy import DemoParser
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

failed = list()


def _parse_file(f: str):
  """Parses a demofile at a given filepath using the awpy library and stores it as a structured json

  Args:
      f (str): filepath of a demofile
  """
  try:
    demo_parser = DemoParser(
        demofile=f, outpath='./data/interim', json_indentation=True, log=False, buy_style="csgo")
    demo_parser.parse()
  except:
    failed.append(f)


def _parse_files():
  """Gathers all raw demofiles as a list and utilizes multithreading to parse them
  """
  files = glob.glob('./data/raw/*.dem')
  #Ensure at least one core
  cpu_count = os.cpu_count()
  cpu_count -= 1
  if cpu_count <= 1:
    cpu_count = 1

  pool = mp.Pool(processes=cpu_count)
  for _ in tqdm(pool.imap_unordered(_parse_file, files), total=len(files)):
      pass


def main():
  """ Runs data processing scripts to parse the raw demofiles into structured json data
  """
  _start = time.time()
  logger = logging.getLogger(__name__)
  logger.info('Parsing demofiles ...')
  _parse_files()
  _end = time.time()
  
  if len(failed) > 0:
    logger.info('==================================')
    logger.error(f"❌ These files failed to parse: {failed}")
    logger.info('==================================')
    

  logger.info('==================================')
  logger.info('Parsing DONE ✅')
  logger.info('---------------')
  logger.info(f'took {_end-_start} second(s)')
  


if __name__ == '__main__':
  log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.INFO, format=log_fmt)

  main()
