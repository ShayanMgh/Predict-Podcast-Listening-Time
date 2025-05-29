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

# Basic information about the dataset
def explore_data(df, name):
    print(f"\n=== {name} Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
explore_data(train, "Train")
explore_data(test, "Test")

# Target variable analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(train['Listening_Time_minutes'], bins=50, alpha=0.7)
plt.title('Distribution of Listening Time')
plt.xlabel('Listening Time (minutes)')

plt.subplot(1, 3, 2)
plt.hist(np.log1p(train['Listening_Time_minutes']), bins=50, alpha=0.7)
plt.title('Log Distribution of Listening Time')
plt.xlabel('Log(Listening Time + 1)')

plt.subplot(1, 3, 3)
plt.boxplot(train['Listening_Time_minutes'])
plt.title('Boxplot of Listening Time')
plt.ylabel('Listening Time (minutes)')

plt.tight_layout()
plt.show()

# Correlation analysis
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
