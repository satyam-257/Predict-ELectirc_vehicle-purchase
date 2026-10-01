# %% [markdown] Cell 0
Hello Everyone! 

Below is the THE COMPLETE ORIGINAL GENERATOR SCRIPT (generate_ev_dataset.py) 
Best of luck for the competition ❤️

# %% Cell 1
import random
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(101)
random.seed(101)

n_buyers = 10000

# 1. Basic Demographics
buyer_id = [f"EV{str(i).zfill(5)}" for i in range(1, n_buyers + 1)]
age = np.random.randint(25, 70, size=n_buyers)
gender = np.random.choice(
    ['Male', 'Female', 'Other'], p=[0.52, 0.45, 0.03], size=n_buyers
)
annual_income = np.random.normal(85000, 35000, size=n_buyers)
annual_income = np.clip(annual_income, 30000, 250000).astype(int)

# 2. Geography & Car Ownership
city_type = np.random.choice(
    ['Urban', 'Suburban', 'Rural'], p=[0.5, 0.35, 0.15], size=n_buyers
)
num_cars_owned = np.random.choice(
    [1, 2, 3, 4], p=[0.4, 0.4, 0.15, 0.05], size=n_buyers
)
current_car_type = np.random.choice(
    ['Sedan', 'SUV', 'Hatchback', 'Truck'],
    p=[0.4, 0.35, 0.15, 0.1],
    size=n_buyers,
)

# 3. Commute & Infrastructure
daily_commute_km = np.random.normal(40, 25, size=n_buyers)
daily_commute_km = np.clip(daily_commute_km, 5, 200).round(1)

charging_home = []
charging_work = []
home_charging_possible = []

for city in city_type:
  if city == 'Urban':
    charging_home.append(np.random.randint(2, 15))
    charging_work.append(np.random.randint(3, 20))
    home_charging_possible.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
  elif city == 'Suburban':
    charging_home.append(np.random.randint(0, 8))
    charging_work.append(np.random.randint(1, 10))
    home_charging_possible.append(np.random.choice(['Yes', 'No'], p=[0.8, 0.2]))
  else:  # Rural
    charging_home.append(np.random.randint(0, 3))
    charging_work.append(np.random.randint(0, 4))
    home_charging_possible.append(np.random.choice(['Yes', 'No'], p=[0.9, 0.1]))

# 4. Psychology & Policies
environmental_concern = np.random.randint(1, 6, size=n_buyers)
subsidy_available = np.random.choice(['Yes', 'No'], p=[0.6, 0.4], size=n_buyers)

# 5. Target Variable 1: Range Anxiety
anxiety_score = (daily_commute_km / 200.0) * 0.4
anxiety_score += np.where(np.array(home_charging_possible) == 'No', 0.3, 0.0)
anxiety_score -= (np.array(charging_home) + np.array(charging_work)) * 0.01

range_anxiety_level = []
for score in anxiety_score:
  score += np.random.normal(0, 0.1)
  if score > 0.45:
    range_anxiety_level.append('High')
  elif score > 0.25:
    range_anxiety_level.append('Medium')
  else:
    range_anxiety_level.append('Low')

# 6. Target Variable 2: EV Purchase Intent
buy_score = (annual_income / 250000.0) * 0.3 + (environmental_concern / 5.0) * 0.3
buy_score += np.where(np.array(subsidy_available) == 'Yes', 0.2, 0.0)
buy_score -= np.where(np.array(range_anxiety_level) == 'High', 0.3, 0.0)
buy_score -= np.where(np.array(range_anxiety_level) == 'Medium', 0.1, 0.0)

will_buy_ev = []
for score in buy_score:
  score += np.random.normal(0, 0.1)
  if score > 0.55:
    will_buy_ev.append('Yes')
  else:
    will_buy_ev.append('No')

# 7. Assemble DataFrame
df = pd.DataFrame({
    'Buyer_ID': buyer_id,
    'Age': age,
    'Gender': gender,
    'Annual_Income_USD': annual_income,
    'City_Type': city_type,
    'Daily_Commute_km': daily_commute_km,
    'Number_of_Cars_Owned': num_cars_owned,
    'Current_Car_Type': current_car_type,
    'Charging_Stations_Near_Home': charging_home,
    'Charging_Stations_Near_Work': charging_work,
    'Home_Charging_Possible': home_charging_possible,
    'Environmental_Concern_Level': environmental_concern,
    'Subsidy_Available': subsidy_available,
    'Range_Anxiety_Level': range_anxiety_level,
    'Will_Buy_EV': will_buy_ev,
})

# 8. Missing Values Injection (~2%)
for col in [
    'Annual_Income_USD',
    'Daily_Commute_km',
    'Environmental_Concern_Level',
]:
  mask = np.random.rand(n_buyers) < 0.02
  df.loc[mask, col] = np.nan

# 9. Save Dataset
df.to_csv('EV_Adoption_and_Range_Anxiety_Dataset.csv', index=False)

print(f"Shape: {df.shape}")
display(df.head())

