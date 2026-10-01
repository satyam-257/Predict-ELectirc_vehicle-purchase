import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
orig = pd.read_csv('./data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv')

feature_cols = ['Age', 'Gender', 'Annual_Income_USD', 'City_Type', 'Daily_Commute_km', 
                'Number_of_Cars_Owned', 'Current_Car_Type', 'Charging_Stations_Near_Home', 
                'Charging_Stations_Near_Work', 'Home_Charging_Possible', 
                'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level']

print(f"Original rows: {len(orig)}")
train_orig = pd.merge(train, orig, on=feature_cols, how='inner')
print(f"Train exact matches with Original: {len(train_orig)}")

test_orig = pd.merge(test, orig, on=feature_cols, how='inner')
print(f"Test exact matches with Original: {len(test_orig)}")

# Check target consistency on train matches
if len(train_orig) > 0:
    target_match = (train_orig['Will_Buy_EV_x'] == train_orig['Will_Buy_EV_y']).mean()
    print(f"Target match consistency on Train: {target_match:.4f}")

# Check rounding / float precision matches
print("\nChecking with rounded numericals (Annual_Income, Daily_Commute):")
orig_rounded = orig.copy()
orig_rounded['Annual_Income_USD'] = orig_rounded['Annual_Income_USD'].round(0)
orig_rounded['Daily_Commute_km'] = orig_rounded['Daily_Commute_km'].round(1)

test_rounded = test.copy()
test_rounded['Annual_Income_USD'] = test_rounded['Annual_Income_USD'].round(0)
test_rounded['Daily_Commute_km'] = test_rounded['Daily_Commute_km'].round(1)

test_orig_round = pd.merge(test_rounded, orig_rounded, on=feature_cols, how='inner')
print(f"Test matches with rounding: {len(test_orig_round)}")
