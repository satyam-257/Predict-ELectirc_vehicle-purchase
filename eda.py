import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print("="*50)
print("DATA OVERVIEW")
print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")
print("Target Distribution:")
print(train['Will_Buy_EV'].value_counts(normalize=True))

print("\n" + "="*50)
print("NUMERICAL FEATURES STATS (TRAIN)")
num_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
print(train[num_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n" + "="*50)
print("NUMERICAL FEATURES STATS (TEST)")
print(test[num_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n" + "="*50)
print("CATEGORICAL FEATURES VALUE COUNTS & TARGET RATE")
cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
for col in cat_cols:
    summary = train.groupby(col)['Will_Buy_EV'].agg(
        Count='count',
        Buy_Rate=lambda x: (x == 'Yes').mean()
    )
    print(f"\n--- {col} ---")
    print(summary)

print("\n" + "="*50)
print("CHECKING FOR DUPLICATES")
feature_cols = [c for c in test.columns if c != 'id']
print(f"Train duplicate rows: {train.duplicated(subset=feature_cols).sum()}")
print(f"Test duplicate rows: {test.duplicated(subset=feature_cols).sum()}")

print("\n" + "="*50)
print("CORRELATION WITH TARGET (Numerical)")
train_numeric = train[num_cols].copy()
train_numeric['Target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
print(train_numeric.corr()['Target'].sort_values(ascending=False))
