# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train = pd.read_csv('/Users/shine/Desktop/Kaggle/Predict Podcast Listening Time/playground-series-s5e4/train.csv')
test = pd.read_csv('/Users/shine/Desktop/Kaggle/Predict Podcast Listening Time/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/Users/shine/Desktop/Kaggle/Predict Podcast Listening Time/playground-series-s5e4/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)