#!/usr/bin/python

"""
Author: Alberto Gil Jimenez
Course: Data Mining Techniques


Inspect data
"""

# Import desired libraries
import random
import os
import matplotlib
import pylab as plt
from operator import itemgetter
import numpy
from datetime import datetime, date
import time
import pandas
from random import randint

dataset = 'data/dataset_mood_smartphone.csv'

def retrieve_individual_list(inputfile):
    # Import the raw dataset into a pandas "data-frame" (structure similar to df in R)
    # How to work with data-frames -> Check cheat sheet
    dataset = pandas.read_csv(inputfile)
    return dataset.id.unique().tolist()

# Read csv file, 2nd version
# First approah: retrieve/preprocess the data of just one individual ID: AS14.01
def read_file_v2(individual,inputfile,twindow):
    # Import the raw dataset into a pandas "data-frame" (structure similar to df in R)
    # How to work with data-frames -> Check cheat sheet
    dataset = pandas.read_csv(inputfile)

    ### MISCELLANEOUS ##
    # TIP: logical operators are different in pandas. "&" is and, "|" is or
    # Set next variable to True if you want to see examples
    printscreen = False
    if printscreen:
        # The column values of the dataset can be checked by:
        #     (In this case they are: ['Unnamed: 0' 'id' 'time' 'variable' 'value'])
        print dataset.columns.values
        # If we want to access specific rows of the data frame: (e.g. all the rows which have variable = mood)
        print dataset[dataset['variable'] == "mood"]
        # We can even do numeric operations with numpy on a data frame. E.g.
        print numpy.mean(dataset[dataset['id'] == 'AS14.02']['value'])
        # If we want to check object types (documentation in google/pandas website)
        print type(dataset['time'].dt.month)

    ### end of MISCELLANEOUS ###

    # Delete the column 'Unnamed: 0 '(contains useless information)
    del dataset['Unnamed: 0']
    #print dataset.columns.values
    # Convert the time elements into a "time-readable" object (time format python)
    dataset['time'] = pandas.to_datetime(dataset['time'], coerce=True)

    # Create new column for storing the time in a different way. daydiff === day difference
    dataset['daydiff'] = 0
    # dataset.id.unique() will give you a list of different 'id' values
    #     in column 'id' (but without repentitions <-> non-redundant)
    for individual_in_list in dataset.id.unique():
        # Compute the time difference between the day of the stored record (e.g. 2014-04-30 14:24:03.618)
        # and the first time day (min) in which a smartphone value was recorded.
        minday = min(dataset[dataset['id']==individual_in_list]['time'])
        g = dataset[dataset['id']==individual_in_list]['time'] - minday
        # Store this time difference in the new column ('daydiff'), only in DAYS
        #    e.g. now 2014-04-30 14:24:03.618 - 2014-04-18 17:29:03.618 is 22 days!
        dataset.loc[dataset['id']==individual_in_list, 'daydiff'] = g.dt.days
    # As we're no longer interested in the recorded date, delete the time column
    del dataset['time']
    # For simplicity we'll work only with the data from one individual
    newdataset = dataset[dataset['id'] == individual]

    #print newdataset.groupby(['daydiff'], ).mean()
    ndataset_mean = newdataset[(newdataset['variable']=='activity') | (newdataset['variable']=='mood')| (newdataset['variable']=='circumplex.arousal') | (newdataset['variable']=='circumplex.valence')]
    mean_grouped = ndataset_mean.groupby(['daydiff','variable'], as_index = False ).mean()
    #print mean_grouped

    # this works (UNCOMMENT)
    ndataset_sum = newdataset[(newdataset['variable']!='activity') & (newdataset['variable']!='mood') & (newdataset['variable']!='circumplex.arousal') & (newdataset['variable']!='circumplex.valence')]
    sum_grouped = ndataset_sum.groupby(['daydiff','variable'], as_index = False ).sum()
    #print sum_grouped

    globaldf = pandas.concat([mean_grouped,sum_grouped])
    #print globaldf

    timewindow = twindow
    maxday = max(globaldf['daydiff'])
    # print type(maxday)

    # List the variables in the dataset
    variables = dataset.variable.unique()
    target_variable = 'mood'




    ntest = pandas.DataFrame(columns=mean_grouped.columns.values)
    n1 = mean_grouped[mean_grouped['daydiff']==74]
    n2 = mean_grouped[mean_grouped['daydiff']==89]

    ntttt = pandas.concat([ntest,n1])
    nt = pandas.concat([n1,n2])
    nt = nt.groupby(['variable'], as_index = False ).mean()
    del nt['daydiff']
    #print nt
    #print nt.empty # for checking if the data frame is empty
    ## This is the value that we want!!!

    #print nt[nt['variable']=='activity']['value'].get_value(0,0)

    ndframe = pandas.DataFrame(columns=variables)
    # test things on this dataframe:
    #print ndframe
    ndframe.loc[0] = ['NA' for n in range(19)]
    ndframe.loc[1] = [1 for n in range(19)]

    #print ndframe
    #print '\n'
    # this replaces the second row of the mood column with this value
    ndframe['mood'][1] = 'NA'
    #    print ndframe['mood'][1]
    #print ndframe

    # Create a new dataframe for storing all the info from the variables
    # on the selected time window
    columns_list  = variables.tolist() + ['targetmood'] + ['benchmark']
    ndframe = pandas.DataFrame(columns=columns_list)

    # Create a DF for storing the info, but without time windows (to be used by ARIMA)
    columns_list = variables.tolist() + ['day']
    ndframe_notimewindow = pandas.DataFrame(columns=columns_list)


    # Assign the window lenght of the temporal subsets
    timewindow = twindow
    # Create a variable for storing the number of assigned rows / temporal subsets
    assigned = 0
    # Iterate through all the possible starting days of the possible timewindows
    #    e.g. If we have records on days 0,1,2,3,4,5,6,7, with a window length of
    #    3 the possible starting days are 0,1,2,3,4,5
    for days in range(0,maxday-timewindow+2):
        # Create empty dataframe for concatenating other data frames,
        # for those sub-data frames which contained variables that will be AVERAGED in the time window
        iterabledf = pandas.DataFrame(columns=mean_grouped.columns.values)
        # Create empty dataframe for concatenating other data frames,
        # for those sub-data frames which contained variables that will be SUMMED in the time window
        iterabledf_sum = pandas.DataFrame(columns=sum_grouped.columns.values)
        variables_0 = []
        #print iterabledf.variable.unique()
        #print ndframe
        for var0 in iterabledf_sum.variable.unique():
            variables_0.append(ndframe.columns.get_loc(var0))
        # Append a new empty row in the final data frame
        #l = [numpy.nan for n in range(19)]
        l = [numpy.nan for n in range(3)] + [0 for n in range(3,21)]
        m = [numpy.nan for n in range(3)] + [0 for n in range(3,20)]
        #l = numpy.array(l)
        #l[variables_0] = 0
        #ndframe.loc[assigned] = ['NA' for n in range(19)]
        ndframe.loc[assigned] = l
        ndframe_notimewindow.loc[assigned] = m


        # Iterate through all the days contained in that specific window length.
        #    e.g. if we are in day=3 and the window lenght is 3, we'll Iterate
        #     through days 3,4,5
        for day in range(days,days+timewindow):
            # retrieve info from temporal window, and append it in the "iterabledf" dataframe
            temporaldf = mean_grouped[mean_grouped['daydiff']==day]
            iterabledf = pandas.concat([iterabledf,temporaldf])
            # retrieve info from temporal window, and append it in the "iterabledf_sum" dataframe
            temporaldf_sum = sum_grouped[sum_grouped['daydiff']==day]
            iterabledf_sum = pandas.concat([iterabledf_sum,temporaldf_sum])
            # Store the mood in the target_mood variable
            # (this will enable, once we're out of the loop, to retrieve the mood
            #      of the last day, which corresponds to our target mood value)
            if day == days+timewindow-1:
                # mood of the next day
                temporaldf = mean_grouped[mean_grouped['daydiff']==day + 1]
                # mood of the last day
                temporaldf_benchmark = mean_grouped[mean_grouped['daydiff']==day]
                if temporaldf[temporaldf['variable']=='mood'].index.tolist() != []:
                    index_row_tf = temporaldf[temporaldf['variable']=='mood'].index.tolist()[0]
                    target_mood =  temporaldf[temporaldf['variable']=='mood']['value'].get_value(index_row_tf,0)
                # If the mood of the last day is missing, store a 'NaN' object
                else:
                    target_mood = numpy.nan
                if temporaldf_benchmark[temporaldf_benchmark['variable']=='mood'].index.tolist() != []:
                    index_row_tf = temporaldf_benchmark[temporaldf_benchmark['variable']=='mood'].index.tolist()[0]
                    mood_benchmark =  temporaldf_benchmark[temporaldf_benchmark['variable']=='mood']['value'].get_value(index_row_tf,0)
                # If the mood of the last day is missing, store a 'NaN' object
                else:
                    mood_benchmark = numpy.nan

        # Retrieve the info for storing the info without temporal windows (for building
        # the DF that will be used in ARIMA)
        temporaldf_nogrouped = mean_grouped[mean_grouped['daydiff']==days]
        temporaldf_sum_nogrouped = sum_grouped[sum_grouped['daydiff']==days]

        # Do the mean over the whole time window, for every variable
        iterabledf = iterabledf.groupby(['variable'], as_index = False ).mean()
        # Do the sum over the whole time window, for every variable
        iterabledf_sum = iterabledf_sum.groupby(['variable'], as_index = False ).sum()
        # Do the mean over the time day, for every variable
        temporaldf_nogrouped = temporaldf_nogrouped.groupby(['variable'], as_index = False ).mean()
        # Do the sum over the whole day, for every variable
        temporaldf_sum_nogrouped = temporaldf_sum_nogrouped.groupby(['variable'], as_index = False ).sum()

        # Iterate through all variables that were AVERAGED
        for variable in iterabledf.variable.unique():

            # Check if the variable info contained in that data frame is empty
            # (in this case, this means lack of data for that variable/time window)
            if iterabledf[iterabledf['variable']==variable]['value'].empty:
                # assign NA. if variable is empty
                ndframe[variable][ndframe] = numpy.nan
            else:
                s = iterabledf[iterabledf['variable']==variable]['value']
                # Retrieve the row index from the iterabledf data frame for that specific variable
                index_row_var = iterabledf[iterabledf['variable']==variable].index.tolist()[0]
                # Retrieve the mean value of that variable over the selected window lenght
                meanvalue = iterabledf[iterabledf['variable']==variable]['value'].get_value(index_row_var,0)
                # assign this new value in the variable column, in row "assigned"
                ndframe[variable][assigned] = meanvalue
        # (Do the same of the previous loop, but for all variables that were SUMMED)
        ### this following loop is not working!! variables are not accesses well (wrong list )
        for variable in iterabledf_sum.variable.unique():
            # Check if the variable info contained in that data frame is empty
            # (in this case, this means lack of data for that variable/time window)
            statement = iterabledf_sum[iterabledf_sum['variable']==variable]['value'].empty
            #print statement
            if iterabledf_sum[iterabledf_sum['variable']==variable]['value'].empty:
                # assign null value if variable is empty
                ndframe[variable][ndframe] = 0
                #print 'imhere'
            else:
                s = iterabledf_sum[iterabledf_sum['variable']==variable]['value']
                # Retrieve the row index from the iterabledf data frame for that specific variable
                index_row_var = iterabledf_sum[iterabledf_sum['variable']==variable].index.tolist()[0]
                # Retrieve the summd value of that variable over the selected window lenght
                sumvalue = iterabledf_sum[iterabledf_sum['variable']==variable]['value'].get_value(index_row_var,0)
                # assign this new value in the variable column, in row "assigned"
                ndframe[variable][assigned] = sumvalue
        # Assign the target mood (mood of the next day of the time window)
        ndframe['targetmood'][assigned] = target_mood
        # Assign the mood of the last day of the time window (benchmark)
        ndframe['benchmark'][assigned] = mood_benchmark
#####################################
    #   # Do the same of the two previous loops, but for the data frame that will be used in ARIMA
        # Iterate through all variables that were AVERAGED
        for variable in temporaldf_nogrouped.variable.unique():
            # Check if the variable info contained in that data frame is empty
            # (in this case, this means lack of data for that variable/time window)
            if temporaldf_nogrouped[temporaldf_nogrouped['variable']==variable]['value'].empty:
                # assign NA. if variable is empty
                ndframe_notimewindow[variable][ndframe_notimewindow] = numpy.nan
            else:
                s = temporaldf_nogrouped[temporaldf_nogrouped['variable']==variable]['value']
                # Retrieve the row index from the temporaldf_nogrouped data frame for that specific variable
                index_row_var = temporaldf_nogrouped[temporaldf_nogrouped['variable']==variable].index.tolist()[0]
                # Retrieve the mean value of that variable over the selected window lenght
                meanvalue = temporaldf_nogrouped[temporaldf_nogrouped['variable']==variable]['value'].get_value(index_row_var,0)
                # assign this new value in the variable column, in row "assigned"
                ndframe_notimewindow[variable][assigned] = meanvalue
        # (Do the same of the previous loop, but for all variables that were SUMMED)
        ### this following loop is not working!! variables are not accesses well (wrong list )
        for variable in temporaldf_sum_nogrouped.variable.unique():
            # Check if the variable info contained in that data frame is empty
            # (in this case, this means lack of data for that variable/time window)
            statement = temporaldf_sum_nogrouped[temporaldf_sum_nogrouped['variable']==variable]['value'].empty
            #print statement
            if temporaldf_sum_nogrouped[temporaldf_sum_nogrouped['variable']==variable]['value'].empty:
                # assign null value if variable is empty
                ndframe_notimewindow[variable][ndframe_notimewindow] = 0
                #print 'imhere'
            else:
                s = temporaldf_sum_nogrouped[temporaldf_sum_nogrouped['variable']==variable]['value']
                # Retrieve the row index from the iterabledf data frame for that specific variable
                index_row_var = temporaldf_sum_nogrouped[temporaldf_sum_nogrouped['variable']==variable].index.tolist()[0]
                # Retrieve the summd value of that variable over the selected window lenght
                sumvalue = temporaldf_sum_nogrouped[temporaldf_sum_nogrouped['variable']==variable]['value'].get_value(index_row_var,0)
                # assign this new value in the variable column, in row "assigned"
                ndframe_notimewindow[variable][assigned] = sumvalue
        ndframe_notimewindow['day'][assigned] = days
###########################
        # Increment by 1 the number of assigned rows
        assigned += 1


    ndframe_withoutNA = ndframe.dropna()
    ndframe_notimewindow_withoutNA = ndframe_notimewindow.dropna()

    # Return the dataframe with each variable averaged per time window
    return ndframe, ndframe_withoutNA, ndframe_notimewindow_withoutNA, ndframe_notimewindow


def main():
    individuals_list = retrieve_individual_list(dataset)
    # Select a time window
    timewindow = 1
    # DICTIONARIES STORING DATA. Keys: individual. Values: data frames
    # Create dictionary for storing the raw data frame (with NA) per individual
    ndframe = dict()
    # Create dictionary for storing the data frame (without NA) per individual
    ndframe_withoutNA = dict()
    # Create dictionary for storing the raw data frame (with NA) per individual,
    # but without using time windows (to be used by ARIMA)
    ndframe_notimewindow = dict()
    # Create dictionary for storing the data frame (without NA) per individual,
    # but without using time windows (to be used by ARIMA)
    ndframe_notimewindow_withoutNA = dict()

    n=0
    for individual in individuals_list:
        n+=1
        print n
        ndframe_temp, ndframe_withoutNA_temp, ndframe_notimewindow_withoutNA_temp, ndframe_notimewindow_temp = read_file_v2(individual,dataset,timewindow)
        ndframe[individual] = ndframe_temp
        ndframe_withoutNA[individual] = ndframe_withoutNA_temp
        ndframe_notimewindow[individual] = ndframe_notimewindow_withoutNA_temp
        ndframe_notimewindow_withoutNA[individual] = ndframe_notimewindow_temp
    return ndframe, ndframe_withoutNA, ndframe_notimewindow, ndframe_notimewindow_withoutNA, timewindow, individuals_list


if __name__=="__main__":
    main()
