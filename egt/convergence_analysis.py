"""Visualize convergence"""
import numpy as np
import matplotlib.pyplot as plt


def func(history, f):
    location_hist = [locs for locs, strats in history]
    print(location_hist)