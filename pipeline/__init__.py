import logging
import sys
from os.path import dirname, realpath

src_path = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, src_path)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
