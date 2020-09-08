from dataset.ml_dataset import *
import numpy as np

class RandomForestEstimator:
    def __init__(self):
        self.data = MLDataset()
        self.data.load()