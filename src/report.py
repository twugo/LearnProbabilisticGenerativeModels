# coding: utf-8

# writer: Takuya Togo

import os
import numpy as np
import matplotlib.pyplot as plt

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

class Report:
    def q1(self):
        pass

def main():
    report = Report()
    report.q1()


if __name__ == '__main__':
    pass
