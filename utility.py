import os
import json
import pickle
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def write_file(file_path, data_list):
    """
        This function writes the pickled representation of the data into the provided file path.
        Args:
            file_path: full path of the data file
            data_list: data to be pickled
    """
    ## Pickle the data and dump it at the provided path in binary format.
    with open(file_path, 'wb') as handle:
        pickle.dump(data_list, handle, protocol = 2)

def read_file(file_path, file_type):
    """
        This function reads the data from the provided file path and returns the read data.
        Args:
            file_path: full path of the data file
            file_type: type of file to be read
                0: json file
        Returns:
            
    """
    ## Check the type of file and read the data accordingly.
    if (file_type == 0):
        with open(file_path, 'r') as file:
            data = json.load(file)
    
    ## Return the read data.
    return data

def current_time():
    """
        This function prints the current system time. 
    """
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)