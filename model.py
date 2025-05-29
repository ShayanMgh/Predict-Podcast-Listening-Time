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

def feature_engineering(df):
    """
    Perform feature engineering on the dataset
    """
    df = df.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove id and target from numerical columns if present
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if 'Listening_Time_minutes' in numerical_cols:
        numerical_cols.remove('Listening_Time_minutes')
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Handle categorical variables with label encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Create interaction features
    if len(numerical_cols) >= 2:
        # Create some polynomial features
        for i, col1 in enumerate(numerical_cols[:3]):  # Limit to avoid too many features
            for col2 in numerical_cols[i+1:4]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
    
    # Create statistical features
    for col in numerical_cols:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    return df, label_encoders

# Apply feature engineering
train_fe, label_encoders = feature_engineering(train)
test_fe, _ = feature_engineering(test)

print(f"Original train shape: {train.shape}")
print(f"Feature engineered train shape: {train_fe.shape}")
