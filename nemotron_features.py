import numpy as np
import pandas as pd

def add_nemotron_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed non-linear interaction features for EV Adoption.
    """
    df = df.copy()
    
    # Non-linear interaction features
    df['Age_squared'] = df['Age'] ** 2
    df['Income_per_car'] = df['Annual_Income_USD'] / (df['Number_of_Cars_Owned'] + 1)
    df['Commute_station_log'] = df['Daily_Commute_km'] * np.log1p(
        df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    )
    
    # Safe categorical to numeric mappings before multiplications
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    range_num = df['Range_Anxiety_Level'].map(range_map).fillna(0).astype(float)
    df['Env_Concern_RangeAnxiety'] = df['Environmental_Concern_Level'] * range_num
    
    home_charge_num = (df['Home_Charging_Possible'] == 'Yes').astype(float)
    subsidy_num = (df['Subsidy_Available'] == 'Yes').astype(float)
    df['HomeCharging_Subsidy'] = home_charge_num * subsidy_num
    
    return df