import pandas as pd
import numpy as np

def engineer_features(train_path='./data/train.csv', test_path='./data/test.csv'):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Target
    train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
    
    n_train = len(train)
    df = pd.concat([train.drop(columns=['Will_Buy_EV', 'target']), test], axis=0).reset_index(drop=True)
    
    # Convenience aliases
    income = df['Annual_Income_USD']
    commute = df['Daily_Commute_km']
    age = df['Age']
    env = df['Environmental_Concern_Level']
    
    print("Step 1/7: Binary & Ordinal Encodings...")
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Range_Anxiety_Num'] = df['Range_Anxiety_Level'].map(range_map).fillna(0).astype(int)
    df['Subsidy_Available_Num'] = (df['Subsidy_Available'] == 'Yes').astype(int)
    df['Home_Charging_Num'] = (df['Home_Charging_Possible'] == 'Yes').astype(int)
    df['Med_Anx'] = (df['Range_Anxiety_Level'] == 'Medium').astype(int)
    df['Hi_Anx'] = (df['Range_Anxiety_Level'] == 'High').astype(int)
    
    print("Step 2/7: Fable 5.1 Recipe + Composite Scores...")
    # Fable 5.1 buy_score (alone gives ~0.9377 AUC)
    df['buy_score'] = (
        1.2 * (income / 100000.0) +
        0.6 * env +
        2.0 * df['Subsidy_Available_Num'] -
        1.0 * df['Med_Anx'] -
        3.0 * df['Hi_Anx']
    )
    df['score_diff_5_5'] = df['buy_score'] - 5.5
    df['above_threshold'] = (df['buy_score'] >= 5.5).astype(int)
    
    # Worry score
    df['worry_score'] = (
        df['Range_Anxiety_Num'] * 0.4 +
        (10.0 - df['Charging_Stations_Near_Home']) * 0.3 +
        (10.0 - df['Charging_Stations_Near_Work']) * 0.3
    )
    
    # Nathan Jacob Hard Veto & Asymmetries
    df['veto_high_anx_sub'] = ((df['Hi_Anx'] == 1) & (df['Subsidy_Available_Num'] == 1)).astype(int)
    df['no_sub'] = (df['Subsidy_Available_Num'] == 0).astype(int)
    df['sub_low_anx'] = ((df['Range_Anxiety_Num'] == 0) & (df['Subsidy_Available_Num'] == 1)).astype(int)
    
    print("Step 3/7: Charging & Financial Ratios...")
    df['Total_Charging_Stations'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    df['Home_Work_Charging_Ratio'] = df['Charging_Stations_Near_Home'] / (df['Charging_Stations_Near_Work'] + 1)
    df['Charging_Per_Commute_Km'] = df['Total_Charging_Stations'] / (commute + 1)
    df['Home_Charging_Density'] = df['Charging_Stations_Near_Home'] / (commute + 1)
    df['Work_Charging_Density'] = df['Charging_Stations_Near_Work'] / (commute + 1)
    
    df['Income_Per_Car'] = income / (df['Number_of_Cars_Owned'] + 1)
    df['Income_Per_Age'] = income / (age + 1)
    df['Income_Per_Commute'] = income / (commute + 1)
    df['Log_Annual_Income'] = np.log1p(income)
    df['Log_Daily_Commute'] = np.log1p(commute)
    
    print("Step 4/7: Interaction Scores (original + new Subsidy_X_EnvConcern family)...")
    # EV Readiness composite
    df['EV_Readiness_Score'] = (
        env * 2.0 +
        df['Subsidy_Available_Num'] * 3.0 +
        df['Home_Charging_Num'] * 1.5 -
        df['Range_Anxiety_Num'] * 2.5
    )
    # Base interactions
    df['Subsidy_X_EnvConcern'] = df['Subsidy_Available_Num'] * env
    df['HomeCharge_X_EnvConcern'] = df['Home_Charging_Num'] * env
    df['Subsidy_X_HomeCharge'] = df['Subsidy_Available_Num'] * df['Home_Charging_Num']
    df['Income_X_EnvConcern'] = (income / 10000.0) * env
    df['Commute_X_RangeAnxiety'] = commute * (df['Range_Anxiety_Num'] + 1)
    df['concern_x_sub'] = env * (df['Subsidy_Available_Num'] + 0.1)
    df['inc_x_sub'] = (income / 10000.0) * (df['Subsidy_Available_Num'] + 0.1)
    
    # New high-impact Subsidy_X_EnvConcern interactions
    df['Subsidy_EnvConcern_Div_Anxiety'] = df['Subsidy_X_EnvConcern'] / (df['Range_Anxiety_Num'] + 1.0)
    df['Subsidy_EnvConcern_LowAnxiety'] = df['Subsidy_X_EnvConcern'] * (df['Range_Anxiety_Num'] == 0).astype(int)
    df['Subsidy_EnvConcern_HiAnxiety_Veto'] = df['Subsidy_X_EnvConcern'] * (df['Range_Anxiety_Num'] == 2).astype(int)
    df['Subsidy_EnvConcern_Income'] = df['Subsidy_X_EnvConcern'] * (income / 100000.0)
    df['Subsidy_EnvConcern_LogIncome'] = df['Subsidy_X_EnvConcern'] * df['Log_Annual_Income']
    df['Subsidy_EnvConcern_IncomePerCar'] = df['Subsidy_X_EnvConcern'] * (df['Income_Per_Car'] / 100000.0)
    df['Subsidy_EnvConcern_HomeCharge'] = df['Subsidy_X_EnvConcern'] * df['Home_Charging_Num']
    df['Subsidy_EnvConcern_Squared'] = df['Subsidy_Available_Num'] * (env ** 2)
    df['Hard_Veto_Flag'] = ((df['Subsidy_Available_Num'] == 0) | (df['Range_Anxiety_Num'] == 2)).astype(int)
    
    print("Step 5/7: Synthetic Generator Digit Decomposition Artifacts...")
    df['inc_mod10'] = income % 10
    df['inc_mod100'] = income % 100
    df['inc_mod1000'] = income % 1000
    df['inc_mod5'] = income % 5
    df['inc_mod50'] = income % 50
    df['inc_is_div10'] = (income % 10 == 0).astype(int)
    df['inc_is_div100'] = (income % 100 == 0).astype(int)
    df['inc_is_div1000'] = (income % 1000 == 0).astype(int)
    df['inc_is_30k'] = (income == 30000.0).astype(int)
    df['inc_digit1'] = (income // 10000).astype(int)
    df['inc_digit2'] = ((income // 1000) % 10).astype(int)
    
    df['com_dec'] = np.round((commute * 10) % 10).astype(int)
    df['com_is_int'] = (commute % 1 == 0).astype(int)
    df['com_is_5'] = (commute == 5.0).astype(int)
    df['com_int'] = commute.astype(int)
    df['com_mod5'] = np.round(commute).astype(int) % 5
    df['com_mod10'] = np.round(commute).astype(int) % 10
    
    df['age_mod10'] = age % 10
    df['age_mod5'] = age % 5
    df['age_decade'] = (age // 10).astype(int)
    
    print("Step 6/7: Categorical Interactions + Frequency Encodings + Group Aggregations...")
    # Categorical interaction strings
    df['City_Car_Interaction'] = df['City_Type'] + '_' + df['Current_Car_Type']
    df['City_Subsidy_Interaction'] = df['City_Type'] + '_' + df['Subsidy_Available']
    df['HomeCharge_RangeAnxiety_Interaction'] = df['Home_Charging_Possible'] + '_' + df['Range_Anxiety_Level']
    df['Gender_Car_Interaction'] = df['Gender'] + '_' + df['Current_Car_Type']
    df['Sub_Anx_Group'] = df['Subsidy_Available'] + '_' + df['Range_Anxiety_Level']
    df['Sub_Env'] = df['Subsidy_Available'] + '_' + env.astype(str)
    df['City_Anx'] = df['City_Type'] + '_' + df['Range_Anxiety_Level']
    
    # Numeric value frequency encodings (generator artifact leakage)
    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']:
        df[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))
    
    # Categorical frequency encodings
    cat_cols_for_freq = [
        'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 
        'Subsidy_Available', 'Range_Anxiety_Level',
        'City_Car_Interaction', 'City_Subsidy_Interaction', 
        'HomeCharge_RangeAnxiety_Interaction', 'Gender_Car_Interaction',
        'Sub_Anx_Group', 'Sub_Env', 'City_Anx'
    ]
    for c in cat_cols_for_freq:
        freq = df[c].value_counts(normalize=True)
        df[f'{c}_freq'] = df[c].map(freq)
    
    # Group Aggregations (unsupervised — safe on concat)
    for grp in [['City_Type', 'Current_Car_Type'], ['City_Type'], ['Current_Car_Type'], ['Range_Anxiety_Level']]:
        grp_name = '_'.join(grp)
        for num_feat in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level', 'Total_Charging_Stations']:
            agg_stats = df.groupby(grp)[num_feat].agg(['mean', 'std']).reset_index()
            agg_stats.columns = grp + [f'{grp_name}_{num_feat}_mean', f'{grp_name}_{num_feat}_std']
            df = df.merge(agg_stats, on=grp, how='left')
            df[f'{num_feat}_diff_{grp_name}_mean'] = df[num_feat] - df[f'{grp_name}_{num_feat}_mean']
            df[f'{num_feat}_ratio_{grp_name}_mean'] = df[num_feat] / (df[f'{grp_name}_{num_feat}_mean'] + 1e-6)
    
    print("Step 7/7: Encoding categoricals for GBDT...")
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_columns:
        df[c] = df[c].astype('category')
        
    print(f"Total features created: {df.shape[1]}")
    
    train_features = df.iloc[:n_train].copy()
    train_features['target'] = train['target'].values
    test_features = df.iloc[n_train:].copy()
    
    return train_features, test_features, cat_columns

if __name__ == '__main__':
    train_feat, test_feat, cat_cols = engineer_features()
    print("Features ready.")
    print("Train shape:", train_feat.shape)
    print("Test shape:", test_feat.shape)
    print("Categorical columns:", cat_cols)
