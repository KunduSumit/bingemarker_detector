import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_gnuplot(dataframe, show):
    """
        This function creates GNUPlot 
        Args:
            dataframe: .
            show:
    """
    dataframe['start'] = pd.to_datetime(dataframe['start'])
    dataframe['end'] = pd.to_datetime(dataframe['end'])

    cap, start, end = dataframe['episode'], dataframe['start'], dataframe['end']

    #Check the type, because we paint all lines with the same color together
    recap = (dataframe['type'] == 'Recap')
    intro = (dataframe['type'] == 'Intro')
    end_credits = (dataframe['type'] == 'End Credits')

    #Get unique captions and there indices and the inverse mapping
    captions, unique_idx, caption_inv = np.unique(cap, 1, 1)

    #Build y values from the number of unique captions.
    y = (caption_inv + 1) / float(len(captions) + 1)

    #Plot recap timeline black, intro timeline red, and end credits timeline blue respectively
    timelines(y[recap], start[recap], end[recap], 'k')
    timelines(y[intro], start[intro], end[intro], 'r')
    timelines(y[end_credits], start[end_credits], end[end_credits], 'b')
    
    #Setup the plot
    ax = plt.gca()
    # fig, ax = plt.subplots()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(SecondLocator(interval=60)) # used to be SecondLocator(0, interval=20)

    #To adjust the xlimits a timedelta is needed.
    delta = (end.max() - start.min())/10

    plt.yticks(y[unique_idx], captions)
    
    plt.ylim(0,1)
    plt.xlim(start.min()-delta, end.max()+delta)
    
    plt.xlabel('Time')
    plt.ylabel('Episode')
    # plt.title('Show: ', show)
    
    # fig.autofmt_xdate()
    
    plt.show()
    
def timelines(y, xstart, xstop, color='b'):
    """
        This function plots timelines at y from xstart to xstop with given color.
        Args:
            dir_path: path of the directory containing all data files.
        Returns:
            
    """
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)