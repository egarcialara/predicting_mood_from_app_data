#!/usr/bin/python

"""
Author: Elena Garcia
Course: Data Mining Techniques


Plot SVM results
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt




def readfile(filename):
    # list for appending individuals ID, SVM error, and Benchmark error
    ind_list, SVM_error, benchmark_err = [],[],[]
    with open(filename) as fin:
        for line in fin.readlines():
            if line[0] == '#':
                twindow = line.strip('\n')[len(line.strip('\n'))-1]
            else:
                ID = line.strip('\n').split('\t')[0]
                ind_list.append(ID)
                svmerror = line.strip('\n').split('\t')[1]
                SVM_error.append(svmerror)
                dummy_error = line.strip('\n').split('\t')[2]
                benchmark_err.append(dummy_error)

    avg_global_w1 = [str(0.53)]
    bench_glob = [str(0.6287)]

    SVM_error[:0] = avg_global_w1
    benchmark_err[:0] = bench_glob
    ind_list[:0] = ['global model']
    return ind_list, SVM_error, benchmark_err, twindow

def plot_barplot(ind_list, SVM_error, benchmark_err, twindow):

    ind = np.arange(len(ind_list))  # the x locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, SVM_error, width, color='r')
    rects2 = ax.bar(ind + width, benchmark_err, width, color='y')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('RMSE on test set')
    ax.set_title('SVR for time window = %s'%(str(twindow)))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(ind_list,rotation='vertical')
    ax.legend((rects1[0], rects2[0]), ('SVR', 'Benchmark'),fontsize=7,frameon=False)

    plt.tight_layout()
    # plt.show()
    plt.savefig('results/SVR_twindow1.png')


def overall_error(ind_list, RF_error, benchmark_err):
    all_error = 0
    bench_error = 0
    for i in range(0, len(ind_list)):
        all_error += float(RF_error[i])
        bench_error += float(benchmark_err[i])

    mean_error = all_error/len(ind_list)
    mean_bench = bench_error/len(ind_list)
    print 'Mean error:', mean_error, '\nMean benchmarking error:', mean_bench

# Main function
def main():
    # filename = 'results/test.dat'
    filename = 'results/SVR_twindow_1.dat'
    ind_list, SVM_error, benchmark_err, twindow = readfile(filename)
    plot_barplot(ind_list, SVM_error, benchmark_err, twindow)
    overall_error(ind_list, SVM_error, benchmark_err)



if __name__ == "__main__":
    main()
