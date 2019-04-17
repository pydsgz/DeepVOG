# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

class PlotDrawer(object):
    def __init__(self, array_shape):
        self.array_shape = array_shape
    def initiate_plot(self, x, y):
        h, w = self.array_shape
        fig, ax = plt.subplots(figsize=(w/100,h/100))
        