import os
import operator
import pandas as pd
import numpy as np

def get_annotations(data):
    """
    Gets the annotations file as a pandas dataframe.
        data: dict
        
        returns:
            dataframe
    """
    annotations = pd.DataFrame(data).T

    for row, col in annotations.iterrows():
        annotations.loc[row]['start'] = to_seconds(annotations.loc[row]['start'])
        annotations.loc[row]['end'] = to_seconds(annotations.loc[row]['end'])
        
    return annotations

def get_ground_truth(data):
    """
        This function extracts ground truth from the annotation data.
        Args:
            dir_path << directory to feature vector file for each video
            video_list << list of video file names
        Returns:
            
    """
    result = []
    for row, col in data.iterrows():
        result.append((data.loc[row]['start'], data.loc[row]['end']))
        
    return result

def sum_timestamps(timestamps):
    """
    Get the total number of seconds out of a list with timestamps formatted like: (start,end)
    """
    result = 0
    for start,end in timestamps:
        result += end - start
        
    return result

# matches two lists of (starttime,endtime) detections and outputs the relevancy variables
def match_detections_precision_recall(detected, ground_truth, verbose=True):
    """
    Compares the detections and ground truth lists of timestamps and calculates
    and outputs precision and recall scores based on the comparison.
    """
    if verbose:
        print("\nComparing detections with annotations.")
        # print("detected: \t \t {}".format(detected))
        # print("ground truth: \t \t {}".format(ground_truth))


    total_relevant_seconds = sum_timestamps(ground_truth)
    total_detected_seconds = sum_timestamps(detected)
    relevant_detected_seconds = 0    

    for start, end in ground_truth:
        lowest_difference_index = 0
        lowest_difference = -1        

        for i, (start_d, end_d) in enumerate(detected):
  
            if abs(start - start_d) < 2:
                start_d = start
            if abs(end - end_d) < 2:
                end_d = end

            relevant = overlap((start,end), (start_d, end_d))
            relevant_detected_seconds += relevant
    
    if verbose:
        print("Total relevant seconds: {}".format(total_relevant_seconds)) #relevant documents
        print("Total detected seconds: {}".format(total_detected_seconds)) #retrieved documents
        print("Relevant detected seconds: {}".format(relevant_detected_seconds)) #relevant documents AND retrieved documents

        if total_detected_seconds > 0:
            print("Precision = {}".format(relevant_detected_seconds / total_detected_seconds))

        if total_relevant_seconds > 0:
            print("Recall = {}\n".format(relevant_detected_seconds / total_relevant_seconds))

    return total_relevant_seconds, total_detected_seconds, relevant_detected_seconds

def overlap_matching(tup1, tup2):
    """
        This function checks the overlap between time stamps provided. Accepts it if the time difference is less than 2 seconds. Otherwise, rejects it.
        Args:
            tup1: tuple.
            tup2: tuple.
        Returns:
            boolean.
    """
    start, end = tup1
    start_d, end_d = tup2
    
    approved = False
    
    # lowest_difference_index = 0
    # lowest_difference = -1        

    # for i, (start_d, end_d) in enumerate(detected):
  
    if abs(start - start_d) < 2:
        if abs(end - end_d) < 2:
            approved = True
    else:
        if abs(end - end_d) < 2:
            approved = True
    # relevant = overlap((start,end), (start_d, end_d))
    # relevant_detected_seconds += relevant
    
    return approved

def evaluation(dir_path, contents, annotation):
    """
        This function compares the detected contents with accepted annotations for each video file and returns the evaluations stats.
        Args:
            dir_path: directory path.
            contents: detected contents.
            annotation: accepted annotations.
        Returns:
            dictionary.
    """
    all_stats = {}
    video_list = list(contents.keys())
    
    for video_name in video_list:
        # total_relevant_seconds = 0
        # total_detected_seconds = 0
        # total_relevant_detected_seconds = 0
        
        ## Get meta data of the video
        meta_data = read_file(os.path.join(dir_path, video_name.split(".mp4")[0] + '.json'), 0)
        
        ## annotations of the entire data set    
        annotations_list = read_file(os.path.join(dir_path, annotation), 0)
            
        ## all available video file specific annotations 
        video_annotations = list(filter(lambda annotation: annotation['show'] == meta_data['show'] 
                                        and annotation['episode'] == meta_data['episode'] and annotation['path'] == video_name, annotations_list))
        
        print(f"Show: {video_annotations[0]['show']}")
        season = video_annotations[0]['episode'].split("E")[0]
        print(f"Season: {season}")
        episode = video_annotations[0]['episode'].split(season)[1]
        print(f"Episode: {episode}\n")
        
        ## heuristics to select annotation, based on majority approval
        seg = []
        for v_ann in video_annotations:
            seg.append(v_ann['segments']) 
            
        max_dict = max(seg, key=len)
        
        # for dicts in seg:
        #     if (dicts == max_dict):
        #         continue
        #     else:
        #         for key, value in dicts.items():
        #             print(value['type'])
        
        approved_annotation = {}
        approval_threshold = 0.75        
        
        for kd, vd in max_dict.items():
            approval_count = 0
            for s in seg:
                if (s == max_dict):
                    approval_count += 1
                    continue
                else:
                    for key, value in s.items():
                        if (value['type'] == vd['type']):
                            
                            tup1 = (to_seconds(vd['start']), to_seconds(vd['end']))
                            tup2 = (to_seconds(value['start']), to_seconds(value['end']))
                            
                            if (overlap_matching(tup1, tup2)):                     ### check overlap
                                approval_count += 1
            
            approval_rate = approval_count/len(seg)
            
            if (approval_rate >= approval_threshold):
                approved_annotation.update([(kd, vd)])
                
        ## randomly selected annotation
        # annotations = get_annotations(video_annotations[0]['segments'])
        
        annotations = get_annotations(approved_annotation)
        
        # ground_truths = get_skippable_timestamps_by_filename(video, annotations)
        ground_truths = get_ground_truth(annotations)
        relevant_seconds, detected_seconds, relevant_detected_seconds = match_detections_precision_recall(contents[video_name], ground_truths)

        # total_relevant_seconds += relevant_seconds
        # total_detected_seconds += detected_seconds
        # total_relevant_detected_seconds += relevant_detected_seconds
        
        # evaluation(contents[video_name], annotations)
        
        all_stats[video_name] = {'relevant': relevant_seconds, 'detected': detected_seconds, 'relevant_detected': relevant_detected_seconds}
    
    return all_stats
        
def evaluation_metrics(stats):
    """
        This function prints the evaluation metrics provided the evaluation stats.
        Args:
            stats: dictionary.
            
    """
    total_relevant_seconds = 0
    total_detected_seconds = 0
    total_relevant_detected_seconds = 0
    
    for key, value in stats.items():
        total_relevant_seconds += value['relevant']
        total_detected_seconds += value['detected']
        total_relevant_detected_seconds += value['relevant_detected']
    
    precision = total_relevant_detected_seconds / total_detected_seconds
    recall = total_relevant_detected_seconds / total_relevant_seconds

    print("Total precision = {0:.3f}".format(precision))
    print("Total recall = {0:.3f}".format(recall))


    
def pre_process_annotation(dir_path, annotation):
    """
        This function pre-process the annotation file. 
        Args:
            dir_path: path of the directory containing all data files.
        Returns:
            list of accepted annotations
    """
    ## annotations of the entire data set    
    annotations_list = read_file(os.path.join(dir_path, annotation), 0)
    
    accpted_annotations_list = []
    for annotation in annotations_list:
        if annotation['answer'] == 'accept':                                 ## Take only accepted annotations and pass all ignored annotations
            if annotation['segments']:                                      ## take only filled segments
                accpted_annotations_list.append(annotation)
        
    ## remove annotations with hypen issue
    accpted_annotations_list = [a for a in accpted_annotations_list if a['show'] not in ['Star Trek', 'Winning Time', 'Law & Order']]

    return accpted_annotations_list

def annotation_season(accpted_annotations_list):
    """
        This function creates a dictionary of show and its corresponding video file name
        Args:
            dir_path: path of the directory containing all data files.
        Returns:
            dictionary
    """
    ### Videos as per season.
    videos = {}
    show_names = list(map(operator.itemgetter('show'), accpted_annotations_list))
    show_un, show_counts = np.unique(show_names, return_counts=True)

    for show in show_un:
        vals = []
        
        for i, x in enumerate(accpted_annotations_list):
            if x['show'] == show:
                vals.append(x['path'])
        
        videos[show] = vals
    
    return videos

def filter_plot_annotation_show(accpted_annotations_list):
    """
        This function creates gnu plot for each show. 
        Args:
            dir_path: path of the directory containing all data files.
        Returns:
            
    """
    show_names = list(map(operator.itemgetter('show'), accpted_annotations_list))
    show_un, show_counts = np.unique(show_names, return_counts=True)
    
    dataframe = pd.DataFrame()
    
    for show in show_un:
        print("Show Name: ", show)
        
        annotations_list = [a for a in accpted_annotations_list if show == a['show']]
        
        for ann in annotations_list:
            dataframe=pd.concat([dataframe, create_df(ann)])
            
        create_gnuplot(dataframe, show)
            
def create_df(annotation):
    """
        This function creates dataframe 
        Args:
            dir_path: path of the directory containing all data files.
        Returns:
            
    """
    df = pd.DataFrame.from_dict(annotation['segments'], orient='index')
    df['episode'] = annotation['episode']
    
    return df