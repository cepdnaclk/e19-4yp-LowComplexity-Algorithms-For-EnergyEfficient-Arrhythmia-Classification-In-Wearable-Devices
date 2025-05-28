"""
config.py
---------
Configuration class for hyperparameters.
"""

class Config:
    def __init__(self, lr=0.001, batch_size=32, epochs=20):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
