import pandas as pd
df = pd.read_csv('data/train.csv')
print(df[['UltimateIncurredClaimCost', 'WeeklyWages', 'Age']].describe(percentiles=[0.95, 0.98, 0.99]))
