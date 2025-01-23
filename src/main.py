""" 
************************************************************************
 *
 * defines.py
 *
 * Initial Creation:
 *    Author      Nhi Nguyen
 *    Created on  2025-23-01
 *
 ************************************************************************/
"""
from common import defines
from data import make_dataset
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

     make_dataset.read_and_export_to_pickel_file()

  

