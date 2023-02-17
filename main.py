import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import stats
import scipy as sp

logging.basicConfig(level=logging.INFO) # set logging level
plt.style.use('ggplot') # set matplotlib styling
df = pd.read_csv("Placement_Data_Full_Class.csv") # reading the data
