import cv2
import os
import math

import itertools
import operator
import pickle
import faiss
# import concurrent.futures
import datetime

import pandas as pd
import numpy as np

from numba import jit
from concurrent.futures import ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def search_using_faiss(dir_path, video_list):
def search_using_faiss(video_list, vector_list):
    """
        This function extracts 
        Args:
           dir_path:
           video_list:
        Returns:
            
    """
    ## List of full path for the feature vector files.
    # vector_path_list = [os.path.join(dir_path, video.split(".mp4")[0] + '.p') for video in video_list]
    
    ## List of feature vectors.
    # vector_list = []

    # List of lengths of the feature vectors.
    vector_length = [episode_vectors.shape[0] for episode_vectors in vector_list]
    
    ## Iterate through the list of path for all the feature vector files, read the data (feature vectors) from the files and 
    ## append them into a single list.
    # for path in vector_path_list:
    #     episode_vectors = np.array(pickle.load(open(path, "rb")), np.float32)
    #     vector_length.append(episode_vectors.shape[0])
    #     vector_list.append(episode_vectors)
        
    # vector_length = [vector.shape[0] for vector in vector_list]
        
    ## Stack the list of feature vectors vertically (row wise) to form a 
    vectors = np.vstack(vector_list)

    ##
    results = []
    segments = []
    vector_id = []
    vec_epi_indexes = []
    
    ##
    for i, length in enumerate(vector_length):
        print("Querying {}".format(video_list[i]))
        
        i += 1
        s = sum(vector_length[:i-1])
        e = sum(vector_length[:i])

        # query consists of one episode
        query = vectors[s:e]
                
        # rest of the feature vectors
        rest = np.append(vectors[:s], vectors[e:], axis=0)

        # build the faiss index, set vector size
        vector_size = query.shape[1]
        index = faiss.IndexFlatL2(vector_size)

        index.add(rest)
        
        # we want to see k nearest neighbors
        k = 4

        # search with for matches with query
        scores, indexes = index.search(query, k)
        
        ## Get consecutive faiss indexes as segments
        segment_index = []
        all_id_list = []
        cons_id_list = []
        start = -1
        prev = -1

        for i, values in enumerate(indexes):
            if (len(cons_id_list) <= 0): # and prev == -1):
                cons_id_list.append(values[0])
                start = i
                prev = values[0]
            else:
                if (prev == values[0]) or (prev + 1 == values[0]):
                    # if (len(cons_id_list) > 0):
                    #     start = i
                    cons_id_list.append(values[0])
                    prev = values[0]
                else:
                    end = i - 1
                    all_id_list.append(cons_id_list)
                    
                    segment_index.append((start, end))
                    
                    start = i
                    prev = values[0]
                    cons_id_list = []
                    cons_id_list.append(values[0])
                    
            if (i == len(indexes) - 1):
                end = i
                all_id_list.append(cons_id_list)
                 
                segment_index.append((start, end))
         
        ## get vectors indexes and episode indexes for each id
        retrieved_vector_list = []
        
        for id_list in all_id_list:
            un, counts = np.unique(id_list, return_counts = True)
            id_vec_index = {}
            for id in un:
                vec_index = {}
                for i, episode in enumerate(vector_list):
                    # vector_indexes = []
                    # if (rest[[id]] in episode):
                    #     vector_indexes.append(episode.index(rest[[id])) ## index of the vector 
                    
                    vector_indexes  = [j for j, vec in enumerate(episode) if rest[[id]].all() == vec.all()]
                    if (len(vector_indexes) >= 0):
                        vec_index[i] = vector_indexes
                id_vec_index[id] = vec_index
            retrieved_vector_list.append(id_vec_index)

        result = scores[:,0]
        results.append((video_list[i-1], result))
        
        segments.append((video_list[i-1], segment_index))
        vector_id.append((video_list[i-1], all_id_list))
        vec_epi_indexes.append((video_list[i-1], retrieved_vector_list))
        
    ##
    # faiss_result_path = os.path.join(dir_path, "FAISSSearchResult.p")
        
    ## save to pickle file
    # write_file(file_path = faiss_result_path, data_list = results)
    
    ##
    return results, segments, vector_id, vec_epi_indexes

def detect_frame_file(rest, indexes):
    for item in indexes:
        print(item)

def max_two_values(d):
    """ 
    a) create a list of the dict's keys and values; 
    b) return the two keys with the max values
    """  
    v=list(d.values())
    k=list(d.keys())
    result1 = k[v.index(max(v))]
    del d[result1]

    v=list(d.values())
    k=list(d.keys())
    result2 = k[v.index(max(v))]
    return [result1, result2]

def fill_gaps(sequence, lookahead):
    """
    Given a list consisting of 0's and 1's , fills up the gaps between 1's 
    if the gap is smaller than the lookahead.
    Example: 
        input: [0,0,1,0,0,0,0,1,0,0] with lookahead=6
       output: [0,0,1,1,1,1,1,1,0,0]
    """
    
    i = 0
    change_needed = False
    look_left = 0
    to_change = []
    
    while i < len(sequence):
        look_left -= 1
        if change_needed and look_left < 1:
            change_needed = False
        if sequence[i]:
            if change_needed:
                for k in to_change:
                    sequence[k] = True
            else:
                change_needed = True
            look_left = lookahead
            to_change = []
        else:
            if change_needed:
                to_change.append(i)
        i+=1
    return sequence

def get_two_longest_timestamps(timestamps):
    """
    Returns the two longest time intervals given a list of time intervals
    Example: 
        input: [(0,10) , (0,5) , (20,21)]
        returns: [(0,10), (0,5)]
    """
    # if size is smaller or equal to 2, return immediately
    if len(timestamps) <= 2:
        return timestamps

    d = {}
    for start,end in timestamps:
        d[(start,end)] = end - start

    return max_two_values(d)

def to_time_string(seconds):
    """
    Given seconds in integer format, returns a string in the format hh:mm:ss (example: 01:30:45)
    """
    return str(datetime.timedelta(seconds=seconds))

def overlap(interval1, interval2):
    """
    Returns the total amount of overlap between two intervals in the format of (x,y)
    Example:
        input:      (0,10) , (5,10)
        returns:    5  
    """
    return max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))

def merge_consecutive_timestamps(timestamps):
    """
    Merges consecutive timestamps in a list if they're less than 2 seconds apart
    Example: [(0,5), (5,10), (20,30)] gets combined into [(0,10),[20,30]
    """
    result = []
    i = 0
    while i < len(timestamps):
        (start, end) = timestamps[i]

        # check if we're not already at the last element
        if i < len(timestamps) - 1:
            (start_next, end_next) = timestamps[i + 1]
            # merge if less than 2 seconds apart
            if abs(end - start_next) < 2:
                result.append((start, end_next))
                i += 1
            else:
                result.append((start,end))
            
        else:
            result.append((start, end))

        i += 1

    return result

def to_seconds(time):
    """
    Converts string of format hh:mm:ss to total number of seconds
    """
    if time == 'None':
        return -1
    try:
        hours = int(time.split(":")[0])
        minutes = int(time.split(":")[1])
        seconds = int(float(time.split(":")[2]))
        
        return hours*60*60 + minutes*60 + seconds
    except:
        if math.isnan(time):
            return -1

def detect_repeated_contents(dir_path, video_list, percentile=10, framejump=10, video_start_threshold_percentile=20, video_end_threshold_seconds=15, 
                             min_detection_size_seconds=3):
# def detect_repeated_contents(dir_path, video_list, vector_list, percentile=10, framejump=10, video_start_threshold_percentile=20, video_end_threshold_seconds=15, 
#                              min_detection_size_seconds=3):
    """
        This function detects repeated contents.
        Args:
            dir_path: path of the directory containing all data files.
            vector_list: 
            video_list: list of video file names
            percentile:
            framejump:
            video_start_threshold_percentile:
            video_end_threshold_seconds:
            min_detection_size_seconds:
        Returns:
            
    """
    # query the feature vectors of each episode on the other episodes
    results = search_using_faiss(dir_path, video_list)
    # results = search_using_faiss(video_list, vector_list)
    # results, segments, vector_id, vec_epi_indexes = search_using_faiss(video_list, vector_list)
    
    # results = pickle.load(open(os.path.join(dir_path, "FAISSSearchResult_A1_E10_All.p"), "rb"))
    
    all_detections = {}
    # all_stats = {}
    
    ## 
    for video_name, result in results:
        video = cv2.VideoCapture(os.path.join(dir_path, video_name))
        framerate = video.get(cv2.CAP_PROP_FPS)
        
        ## indicates that nearly percentile% of elements of result are lower than threshold
        threshold = np.percentile(result, percentile)

        # all the detections
        below_threshold = result < threshold
        
        # Merge all detections that are less than 10 seconds apart
        below_threshold = fill_gaps(below_threshold, int((framerate/framejump) * 10))

        # put all the indices where values are nonzero in a list of lists
        nonzeros = [[i for i, value in it] for key, it in itertools.groupby(
            enumerate(below_threshold), key=operator.itemgetter(1)) if key != 0]

        detected_beginning = []
        detected_end = []

        # loop through all the detections taking start and endpoint into account
        for nonzero in nonzeros:
            start = nonzero[0]
            end = nonzero[-1]

            #result is in first video_start_threshold% of the video
            occurs_at_beginning = end < len(result) * (video_start_threshold_percentile / 100)
            
            #the end of this timestamp ends in the last video_end_threshold seconds             
            ends_at_the_end = end > len(result) - video_end_threshold_seconds * (framerate/framejump) 

            if (end - start > (min_detection_size_seconds * (framerate / framejump)) #only count detection when larger than min_detection_size_seconds seconds             
                and (occurs_at_beginning or ends_at_the_end)): #only use results that are in first part or end at last seconds          

                start = start / (framerate / framejump)
                end = end / (framerate / framejump)

                if occurs_at_beginning:
                    detected_beginning.append((start,end))
                elif ends_at_the_end:
                    detected_end.append((start,end))


        detected = get_two_longest_timestamps(detected_beginning) + detected_end

        print(f"Detections for: {video_name}")
        
        for start,end in detected:
            
            print("{} \t \t - \t \t {}".format(to_time_string(start), to_time_string(end)))
        print()

    return all_detections
