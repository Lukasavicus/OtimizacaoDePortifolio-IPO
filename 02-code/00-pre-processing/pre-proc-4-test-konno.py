import pandas as pd
from datetime import datetime

df = pd.read_csv('../../01-data/all_assets_hist.csv')

date = df['base_date'].apply(lambda dt : datetime.strptime(dt, '%Y-%m-%d'))
df['date'] = date

year = df['base_date'].apply(lambda dt : int(dt[0:4]))
month = df['base_date'].apply(lambda dt : int(dt[5:7]))

yearmonth = df['base_date'].apply(lambda dt : dt[0:4]+dt[5:7])
df['yearmonth'] = yearmonth

# f_dy = df['dy'] > 0.0

# value_field = 'yield'
value_field = 'dy'

f_y = df[value_field] > 0.0
f_has_dates = df['date'] >= '2021-06-01'
filter_ = f_y & f_has_dates
fdf = df[filter_]

selected_assets = fdf['asset'].unique()#[:100]

f_asset = fdf['asset'].apply(lambda asset: asset in selected_assets)
ffdf = fdf[f_asset]

ffdf_pivoted = ffdf.pivot_table(values = value_field, index = 'yearmonth', columns = 'asset').fillna(0)

print("'",*list(ffdf_pivoted.index), "'", sep="'\t'")

print(*list(selected_assets))

ffdf_pivoted.to_csv('../../01-data/assets_to_konno.txt', header=False, sep='\t')