#!/usr/bin/python

"""
Author: Elena Garcia
Course: Data Mining Techniques


This file plots three graphs in relation
with missing data:

    1- percentage missing data vs participant nr
    2- percentage missing data vs day nr
    3- mood level of participant vs day nr (one per patient)


"""
import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from inspect_data_ARIMA import read_file as df_function
from inspect_data_ARIMA import function2
from inspect_data_ARIMA import retrieve_individual_list as ind_function



def get_indlist():
    global dataset
    dataset = 'data/dataset_mood_smartphone.csv'
    global individuals_list
    individuals_list = ind_function(dataset)


def graph1():
    ''' Plot % missing data vs participant nr
    '''
    mp_list = []
    for individual in individuals_list:
        print individual
        suma = 0
        dataframe = df_function(individual,dataset,1)
        del dataframe['targetmood']
        nan_ind = dataframe.count(axis=0, level=None, numeric_only=False)
        shape = dataframe.shape
        total_ind = shape[0] * shape[1]
        for item in nan_ind:
            suma += item
        nans = total_ind - suma
        perc = nans/float(total_ind)
        mp_list.append(perc)


    y = mp_list
    x = individuals_list

    with open("x and y", "w") as f:
        f.write(str(x) + '\n' + str(y))


    # Plot
    plt.ylim([0, 1])
    plt.xlim([1, len(y)])
    plt.plot(y)
    plt.title('Missing data')
    plt.ylabel('% of missing data'),plt.xlabel('Participant number')
    plt.tight_layout()
    plt.savefig('trial1.png')
    print 'Graph 1 created'

def graph2():
    ''' Plot % missing data vs day nr
    '''

    count_ind = 0
    day_list = [None]*103

    for individual in individuals_list:
        # sum for all individuals
        print individual
        count_ind += 1
        suma = 0
        dataframe = df_function(individual,dataset,1)
        del dataframe['targetmood']
        nan_ind = dataframe.count(axis=1, level=None, numeric_only=False)

        # print len(nan_ind)
        for i in range(0, len(nan_ind)):
            total = 0
            suma = nan_ind[i]
            # put them in a day separated list
            if day_list[i] == None:
                day_list[i] = suma
            else:
                total = day_list[i] + suma
                day_list[i] = total

    shape = dataframe.shape
    total_ind = shape[1] * count_ind

    suma = 0
    list_days = []
    for i in day_list:
        perc = i/float(total_ind)
        perc2 = 1 - perc
        list_days.append(perc2)


    y = list_days
    plt.figure()
    # Plot
    plt.ylim([0, 1])
    plt.plot(y)
    plt.title('Missing data')
    plt.ylabel('% of missing data'),plt.xlabel('Day number')
    plt.tight_layout()
    plt.savefig('trial2.png')
    print 'Graph 2 created'



def graph3():
    ''' Plot the mood level of participant vs day nr (one per patient)'''

    for individual in individuals_list:
        print individual
        # plt.figure()
        globaldf, dataset1 = df_function(individual, dataset)
        dataframe = function2(individual,globaldf,1, dataset1)
        mood_list = dataframe['targetmood'].tolist()
        y = mood_list
        plt.ylim([0, 10])
        # plt.xlim([0, 103])
        plt.xlim([0, len(mood_list)])
        plt.plot(y)
        plt.xlabel('Day number')
        plt.ylabel('Mood')

    plt.title('Mood over time')
    plt.savefig('results/all_individuals_mood.png')
    print 'Graph 3 created'


def main():
    get_indlist()

    # # Use graph1,graph2 and graph3 separately,
    # graph1()
    # graph2()
    graph3()



if __name__=="__main__":
    main()
