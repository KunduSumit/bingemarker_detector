
import cv2
import os
import numpy as np

from numba import jit
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from concurrent.futures import ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_frames(video_path, frames_dir, overwrite, start, end, every):
    """
        This function extracts frames from a video using OpenCV, saves the extraced frames in a directory and returns the list of the extrcted frames.
        Args:
            video_path: full path of the video file
            overwrite: whether to overwrite frames that already exist
            start: start frame
            end: end frame
            every: frame spacing
        Returns:
            List of the frames extracted from the video.
    """
    ## List of all frames in the video file
    frame_list = []
    
    ## Normalize the path names by collapsing redundant separators and up-level references.
    video_path = os.path.normpath(video_path)
    
    ## Get the video directory and the file name from the path.
    video_dir, video_name = os.path.split(video_path)
    
    print("File Name: " + video_name)
    
    ## Check whether the video file exists or not.
    assert os.path.exists(video_path)

    ## Open the video file using OpenCV.
    capture = cv2.VideoCapture(video_path)
    
    ## Start from the first frame, if start frame has not been mentioned.
    if start < 0:
        start = 0
        
    ## Assume the end of the video as the end frame, if end frame has not been mentioned.
    if end < 0:
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ## Set the starting frame as the current frame.
    capture.set(1, start)
    current = start
    
    ## Safety counter to avoid entering an infinite while loop.
    safety = 0
    
    ## Count of frames listed.
    listed = 0
    
    ## Loop through the frames until the end and list them while retaining the sequential order.
    while current < end:
        ## Read the frame.
        _, frame = capture.read()
        
        ## Break, if the safety count gets maxed out.
        if safety > 500:
            break

        ## OpenCV sometimes reads None's during a video. Check whether the frame is a 'None' or a bad flag is returned. If yes, skip. 
        if frame is None:
            ## Increment the safety counter by 1.
            safety += 1
            
            continue
        
        ## Check whether this is the frame to be listed based on the 'every' argument.
        if current % every == 0:
            ## Reset the safety count.
            safety = 0

            ## Resize and add the frame to the list.
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            frame_list.append(resized_frame)
                
            ## Increment the count of listed images by 1.
            listed += 1

        ## Increment the counter by 1 and move to the next frame.
        current += 1
            
    print(f"{listed} frames extracted out of {current} frames.")
        
    ## Close the video file.
    capture.release()
    
    ## Return the list of frames.
    return frame_list

def multiprocessed_feature_extraction(frame_list):
    """
        This function multiprocesses the extraction of feature vectors of the video file.
        Args:
            frame_list: list of frames of the video file
        Returns:
            List of feature vectors.
    """
    print("Extracting features")
    
    # model = VGG16(weights='imagenet', include_top=False)
    
    ## Extract  feature vector for each frame in the list.
    with ThreadPoolExecutor() as executor:
        # pool = executor.map(extract_feature_vector_vgg16, frame_list, model)
        pool = executor.map(extract_feature_vector_vgg16, frame_list)
    
    ## List of feature vector 
    feature_list = []
    
    ## Iterate through the iterator object and append the elements to the feature vector list.
    for res in pool:
        feature_list.append(res)

    ## Return the list of feature vectors.
    return feature_list

model = VGG16(weights='imagenet', include_top=False)

@jit()
# def extract_feature_vector_vgg16(frame, model):
def extract_feature_vector_vgg16(frame):
    """
        This function extracts feature vector of a frame using model based on VGG16.
        Args:
           file_path: full path of the data file
        Returns:
            Array of feature vector.
    """
    ## Expand the dimension of the frame (array) 
    frame_data = np.expand_dims(frame, axis = 0)
    
    ## Preprocess the array encoding the frame using VGG16.
    pre_proc_frame = preprocess_input(frame_data)
    
    ## Predict the feature vector of the preprocessed frame using model based on VGG16.
    vgg16_feature = model.predict(pre_proc_frame)

    ## Return the flattened array of feature vector.
    return vgg16_feature.flatten()
