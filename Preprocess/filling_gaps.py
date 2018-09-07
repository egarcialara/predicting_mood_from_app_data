#!/usr/bin/python

"""
Author: Elena Garcia
Course: Data Mining Techniques


This file finds gaps in the data.
    When it finds a gap of 1-5 in the same patient and variable,
    it fills all the gaps with the average of the values in the sides.
    If the gap is larger, then it chooses only the larger half.

"""

import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


def get_data():
    dataset = open('mock_table','r')
    return dataset

def create_list(dataset):
    ''' This function reads the table and creates one list of lists
    one per participant, with lists of each variable.
    Then, calls the function fill_gaps
    '''
    with open("gap_cuts.txt", "w") as f:
        f.write('\n')

    first = True
    first_for_list = True
    participant = 0
    for line in dataset:
        if first and line.startswith('AS'):
            list_per_participant = []
            first = False
            participant = line.strip('\n')
        elif line.startswith('AS'):
            fill_gaps(list_per_participant, participant, variable_list)
            list_per_participant = []
            first_for_list = True
            participant = line.strip('\n')
        elif line.startswith('[]'):
            line = line.strip('\n')
            line = line.split('\t')
            variable_list = line[1:]
        elif line.startswith('#') or len(line)<2:
            continue
        else:
            line = line.strip('\n')
            line = line.split('\t')
            if first_for_list == True:
                for i in range(1, len(line)):
                    list_per_participant.append([])
                first_for_list = False

            for i in range(1, len(line)):
                if line[i] in ['NA', 0, ' ', '']:
                    line[i] = 'nan'
                list_per_participant[i-1].append(line[i])

    fill_gaps(list_per_participant, participant, variable_list)

def fill_gaps(complete_list, participant, variable_list):
    ''' This function
    1- finds the size of each gap
    2- fills it with the average of the points in the sides
    3- if gap is too big, it chooses the bigger half
        and writes a file stating this.
    '''
    # Delete begining and end gaps
    new_complete_list = []
    for list_ in complete_list:
        for n in range(0, len(list_)):
            first_gap = True
            last_gap = True
            while first_gap == True or last_gap == True:
                if list_[0] == 'nan':
                    list_ = list_[1:]
                else:
                    first_gap = False
                if list_[(len(list_))-1] == 'nan':
                    # print list_
                    list_ = list_[:-1]
                    # print list_
                else:
                    last_gap = False
        new_complete_list.append(list_)
    # print new_complete_list

    complete_list = new_complete_list

    variable = 0
    for list_ in complete_list:
        variable += 1
        nan_found = False
        count_nan = 0
        breakpoint = 0
        for i in range(0, len(list_)):
            if list_[i] == 'nan' and nan_found == False:
                nan_found = True
                count_nan += 1
                last_value = i-1
                gap_checked = False
            elif list_[i] == 'nan':
                nan_found = True
                count_nan += 1
                gap_checked = False
            elif nan_found and list_[i]!='nan':
                new_value = i
                nan_found = False

            # Substitute small gaps with the average
            if nan_found == False and (0<count_nan and count_nan<5) and gap_checked==False: #arbitrary number, change
                avg = (float(list_[new_value]) + float(list_[last_value]))/2
                gap_checked = True
                for j in range(last_value+1, new_value):
                    list_[j] = avg
                last_value = new_value
                count_nan = 0

            # Choose bigger half of data when gap is too big
            if nan_found == False and count_nan>5 and gap_checked==False:
                gap_checked = True
                A = last_value - breakpoint + 1
                B = len(list_) - new_value

                if A<B:
                    for j in range(breakpoint, last_value+1):
                        list_[j] = 'nan'
                else:
                    for j in range(new_value, len(list_)):
                        list_[j] = 'nan'

                # Write in a file these decisions
                start_gap = last_value+1
                end_gap = last_value + count_nan
                variableName = variable_list[variable]
                with open("gap_cuts.txt", "a") as f:
                    f.write('Gap cut in participant ' + str(participant) + ' and variable ' + str(variableName)
                    + '\nIn time points from: ' + str(start_gap) + ' to: ' + str(end_gap) + '\n')

                breakpoint = new_value
                last_value = new_value
                count_nan = 0

    # print complete_list

    # Delete begining and end gaps
    # The function is twice on purpose
    new_complete_list = []
    for list_ in complete_list:
        for n in range(0, len(list_)):
            first_gap = True
            last_gap = True
            while first_gap == True or last_gap == True:
                if list_[0] == 'nan':
                    list_ = list_[1:]
                else:
                    first_gap = False
                if list_[(len(list_))-1] == 'nan':
                    # print list_
                    list_ = list_[:-1]
                    # print list_
                else:
                    last_gap = False
        new_complete_list.append(list_)

    complete_list = new_complete_list
    print complete_list


def main():
    dataset = get_data()
    create_list(dataset)




if __name__=="__main__":
    main()
