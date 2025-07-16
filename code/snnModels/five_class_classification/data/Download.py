#This contains the cod for dwonloading the mit-bih data set into the directory 

import os
import wfdb

DATA_DIR = "data/mitdb"

if os.path.isdir(DATA_DIR):
    print("Dataset already exists.")
      
else : 
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading MIT-BIH data set ......")
    wfdb.dl_database('mitdb', DATA_DIR)
    print("Download ended ! Now you you can work with pre-processing")