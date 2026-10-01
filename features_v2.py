import pandas as pd
import numpy as np
from scipy.stats import norm

def engineer_features_v2(train_path='./data/train.csv', test_path='./data/test.csv'):
    print("Loading raw datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
    n_train = len(train)
    df = pd.concat([train.drop(columns=['Will_Buy_EV', 'target']), test], axis=0).reset_index(drop=True)
    
    print("Step 1/5: Computing base numeric conversions and Fable 5.1 decoded recipe...")
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Range_Num'] = df['Range_Anxiety_Level'].map(range_map).fillna(0).astype(int)
    df['Sub_Num'] = (df['Subsidy_Available'] == 'Yes').astype(int)
    df['Home_Num'] = (df['Home_Charging_Possible'] == 'Yes').astype(int)
    df['Med_Anx'] = (df['Range_Anxiety_Level'] == 'Medium').astype(int)
    df['Hi_Anx'] = (df['Range_Anxiety_Level'] == 'High').astype(int)
    
    # Chris Deotte & Fable 5.1 Recipe (alone gives ~0.9377 AUC)
    income = df['Annual_Income_USD']
    commute = df['Daily_Commute_km']
    age = df['Age']
    env = df['Environmental_Concern_Level']
    
    df['buy_score'] = (
        1.2 * (income / 100000.0) +
        0.6 * env +
        2.0 * df['Sub_Num'] -
        1.0 * df['Med_Anx'] -
        3.0 * df['Hi_Anx']
    )
    df['score_diff_5_5'] = df['buy_score'] - 5.5
    df['above_threshold'] = (df['buy_score'] >= 5.5).astype(int)
    
    # Exact Ground-Truth Generator Features (from Omkar Kadam generator script)
    df['chargers_tot'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    no_home_charging = (df['Home_Charging_Possible'] == 'No').astype(float)
    df['exact_anxiety_score'] = (
        (commute / 200.0) * 0.4 +
        no_home_charging * 0.3 -
        df['chargers_tot'] * 0.01
    )
    # Theoretical normal distribution CDF probabilities for range anxiety
    df['p_high_anxiety'] = 1.0 - norm.cdf((0.45 - df['exact_anxiety_score']) / 0.1)
    df['p_low_anxiety'] = norm.cdf((0.25 - df['exact_anxiety_score']) / 0.1)
    df['p_med_anxiety'] = 1.0 - df['p_high_anxiety'] - df['p_low_anxiety']
    
    # Exact Ground-Truth purchase score and Bayes optimal probability
    df['exact_buy_score'] = (
        (income / 250000.0) * 0.3 +
        (env / 5.0) * 0.3 +
        df['Sub_Num'] * 0.2 -
        df['Hi_Anx'] * 0.3 -
        df['Med_Anx'] * 0.1
    )
    df['exact_prob_will_buy'] = norm.cdf((df['exact_buy_score'] - 0.55) / 0.1)
    df['exact_buy_score_z'] = (df['exact_buy_score'] - 0.55) / 0.1
    
    # Legacy worry score and interactions
    df['worry_score'] = (
        df['Range_Num'] * 0.4 +
        (10.0 - df['Charging_Stations_Near_Home']) * 0.3 +
        (10.0 - df['Charging_Stations_Near_Work']) * 0.3
    )
    df['concern_x_sub'] = env * (df['Sub_Num'] + 0.1)
    df['inc_x_sub'] = (income / 10000.0) * (df['Sub_Num'] + 0.1)
    
    # Nathan Jacob Hard Veto & Asymmetries
    df['veto_high_anx_sub'] = ((df['Hi_Anx'] == 1) & (df['Sub_Num'] == 1)).astype(int)
    df['no_sub'] = (df['Sub_Num'] == 0).astype(int)
    df['sub_low_anx'] = ((df['Range_Num'] == 0) & (df['Sub_Num'] == 1)).astype(int)
    
    print("Step 2/5: Extracting synthetic generator digit decomposition artifacts (+0.0015 OOF)...")
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
    
    print("Step 3/5: Computing frequency encodings across entire generator space...")
    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']:
        df[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))
        
    df['City_Car'] = df['City_Type'] + '_' + df['Current_Car_Type']
    df['City_Sub'] = df['City_Type'] + '_' + df['Subsidy_Available']
    df['Sub_Anx'] = df['Subsidy_Available'] + '_' + df['Range_Anxiety_Level']
    df['Home_Anx'] = df['Home_Charging_Possible'] + '_' + df['Range_Anxiety_Level']
    df['Sub_Env'] = df['Subsidy_Available'] + '_' + df['Environmental_Concern_Level'].astype(str)
    df['City_Anx'] = df['City_Type'] + '_' + df['Range_Anxiety_Level']
    df['Gender_Car'] = df['Gender'] + '_' + df['Current_Car_Type']
    
    pair_cols = [
        'City_Car', 'City_Sub', 'Sub_Anx', 'Home_Anx', 'Sub_Env', 'City_Anx', 'Gender_Car',
        'City_Type', 'Current_Car_Type', 'Range_Anxiety_Level', 'Subsidy_Available', 
        'Home_Charging_Possible', 'Gender'
    ]
    for c in pair_cols:
        df[f'{c}_freq'] = df[c].map(df[c].value_counts(normalize=True))
        
    print("Step 4/5: Computing mobility & financial domain ratios...")
    df['Income_Per_Car'] = income / (df['Number_of_Cars_Owned'] + 1)
    df['Income_Per_Age'] = income / (age + 1)
    df['Income_Per_Commute'] = income / (commute + 1)
    df['Home_Work_Ratio'] = (df['Charging_Stations_Near_Home'] + 1) / (df['Charging_Stations_Near_Work'] + 1)
    df['Charging_Per_Commute'] = df['chargers_tot'] / (commute + 1)
    df['Log_Income'] = np.log1p(income)
    df['Log_Commute'] = np.log1p(commute)
    
    print("Step 5/5: Preparing categorical data types for GBDTs...")
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_columns:
        df[c] = df[c].astype('category')
        
    print(f"Total features engineered: {df.shape[1]}")
    
    train_df = df.iloc[:n_train].copy()
    train_df['target'] = train['target'].values
    test_df = df.iloc[n_train:].copy()
    
    return train_df, test_df, cat_columns

if __name__ == '__main__':
    train_df, test_df, cat_cols = engineer_features_v2()
    print("Train features shape:", train_df.shape)
    print("Test features shape:", test_df.shape)
