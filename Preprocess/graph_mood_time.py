#!/usr/bin/python

"""
Author: Elena Garcia
Course: Data Mining Techniques


This file plots a graphs in relation with patients trends

    plot one graph, with one line per patient
    of mood vs time (day)

"""

import random
import os
import matplotlib
# import pylab as plt
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy
from datetime import datetime, date
import time
import pandas
from random import randint
from inspect_data_ARIMA import main as main_inspect


def plot_graph():
    '''
    1- get the data frame, from Alberto's script: inspectdata.py
    2- get column of days
    3- get column of targetmood
    4- plot
    '''
    ndframe_withoutNA = main_inspect()
    days = ndframe['day'].tolist()
    mood2 = ndframe['mood'].tolist()

    plt.figure()
    plt.ylim(1,10)
    plt.xlim([1, days[len(days)-1]])
    plt.plot(days, mood2)
    plt.show()




def main():
    plot_graph()



if __name__=="__main__":
    main()
