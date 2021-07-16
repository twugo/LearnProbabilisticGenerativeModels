# coding: utf-8

# writer: Takuya Togo

import concurrent.futures
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from ProbabilisticGenerativeModels import *

class Printer:
    @staticmethod
    def plot_figure(data, title, filename = "out.png", xlabel="iteration", ylabel="E"):
        fig = plt.figure()
        plt.title(title)
        # plt.scatter(range(1, len(data)+1), data, s = 1, marker=".", )
        plt.plot(range(1, len(data)+1), data)
        

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        outfolderpath = './out/'
        if not os.path.exists(outfolderpath):
            os.makedirs(outfolderpath)

        plt.savefig(outfolderpath + filename)

    @staticmethod
    def scatter_figure(data, title, filename = "out.png", xlabel="iteration", ylabel="E"):
        fig = plt.figure()
        plt.title(title)
        plt.scatter(range(1, len(data)+1), data, s = 1, marker=".", )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        outfolderpath = './out/'
        if not os.path.exists(outfolderpath):
            os.makedirs(outfolderpath)

        plt.savefig(outfolderpath + filename)

class Solver:
    def make_roc_image(self, n=[3, 3], figtitle='', filename='out.pdf'):
        sigma = 1.0
        input_maker = DefaultInputMaker()

        x_list = []
        y_list = []
        
        x_11_num = 0

        repeat_time = 100000

        for i in range(repeat_time):
            x, y = input_maker.make_dataset(n, sigma)
            x_list.append(x)
            y_list.append(y)

            if((x == np.array([1, 1])).all()):
                x_11_num += 1

        model = ProbabilisticGenerativeModelsForX2()

        theta_list = [i / 100 for i in range(100)]
        data_num = len(theta_list)

        SG_fpr_list = np.zeros(data_num)
        SG_cdr_list = np.zeros(data_num)

        ST_fpr_list = np.zeros(data_num)
        ST_cdr_list = np.zeros(data_num)

        SP12_fpr_list = np.zeros(data_num)
        SP12_cdr_list = np.zeros(data_num)

        theta_index = 0
        for theta in theta_list:
            for i in range(repeat_time):
                x = x_list[i]
                y = y_list[i]

                isx11 = False
                if (x == np.array([1, 1])).all():
                    isx11 = True

                SG = model.calc_S_G(y, n, sigma, input_maker.probability)
                if SG > theta:
                    if isx11:
                        SG_cdr_list[theta_index] += 1
                    else:
                        SG_fpr_list[theta_index] += 1

                ST = model.calc_S_T(y, n, sigma, input_maker.probability)
                if ST > theta:
                    if isx11:
                        ST_cdr_list[theta_index] += 1
                    else:
                        ST_fpr_list[theta_index] += 1

                SP12 = model.calc_S_P_12(y, n, sigma, input_maker.probability)
                if SP12 > theta:
                    if isx11:
                        SP12_cdr_list[theta_index] += 1
                    else:
                        SP12_fpr_list[theta_index] += 1

            theta_index += 1
            print(theta_index)

        SG_cdr_list /= x_11_num
        ST_cdr_list /= x_11_num
        SP12_cdr_list /= x_11_num

        SG_fpr_list /= (repeat_time - x_11_num)
        ST_fpr_list /= (repeat_time - x_11_num)
        SP12_fpr_list /= (repeat_time - x_11_num)


        # plot
        fig = plt.figure()
        plt.title(figtitle)

        plt.xlabel('FPR')
        plt.ylabel('CDR')
        
        # plt.plot(range(1, len(data)+1), data)
        plt.plot(SG_fpr_list, SG_cdr_list, label='SG')
        plt.plot(ST_fpr_list, ST_cdr_list, label='ST')
        plt.plot(SP12_fpr_list, SP12_cdr_list, label='SP12')
        
        plt.legend()

        outfolderpath = './out/'
        if not os.path.exists(outfolderpath):
            os.makedirs(outfolderpath)

        plt.savefig(outfolderpath + filename)

class Report:
    def q1(self):
        n = [3, 3]
        sigma = 1.0

        input_maker = DefaultInputMaker(random.randint(0, 100))
        _, y = input_maker.make_dataset(n, sigma)
        tmp = ProbabilisticGenerativeModelsForX2().calc_S_G(y, n, sigma, input_maker.probability)
        print(tmp)
        tmp = ProbabilisticGenerativeModelsForX2().calc_S_T(y, n, sigma, input_maker.probability)
        print(tmp)
        tmp = ProbabilisticGenerativeModelsForX2().calc_S_P_12(y, n, sigma, input_maker.probability)
        print(tmp)

    def q2(self):
        n = [3, 3]
        filename = 'q2.pdf'
        Solver().make_roc_image(n=n, filename=filename)

    def q3_n55(self):
        n = [5, 5]
        figtitle = 'n1=5,n2=5'
        filename = 'q3n55.pdf'
        Solver().make_roc_image(n=n, figtitle=figtitle, filename=filename)
    
    def q3_n10(self):
        n = [10, 10]
        figtitle = 'n1=10,n2=10'
        filename = 'q3n10.pdf'
        Solver().make_roc_image(n=n, figtitle=figtitle, filename=filename)

    def q3_n20(self):
        n = [20, 20]
        figtitle = 'n1=20,n2=20'
        filename = 'q3n20.pdf'
        Solver().make_roc_image(n=n, figtitle=figtitle, filename=filename)

def main():
    report = Report()
    report.q3_n10()


if __name__ == '__main__':
    main()
