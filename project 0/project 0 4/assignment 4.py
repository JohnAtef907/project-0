import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
df=pd.read_csv('C:\\Users\\LENOVO\\Downloads\\vc studio projects\\python files\\assignment 4\\assignment data.csv')
print(df.head())
print(df.info())
print(df.columns)
print(df.shape)
print(df.isnull().sum().sort_values(ascending=False))
plt.figure(figsize=(4,10))
df.isnull().sum().sort_values(ascending = False).head(6).plot(kind='bar')
plt.ylim(0,df.shape[0])
plt.show()
df.drop("direction_peak_wind_speed",axis = 1,inplace=True)
df.drop("direction_max_wind_speed",axis=1,inplace=True)
df.drop("max_wind_speed",axis=1,inplace=True)
df.drop("days_with_fog",axis =1,inplace=True)
df.drop("energy_star_rating",axis=1,inplace=True)
df["year_built"].fillna(df["year_built"].mean(), inplace=True)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False))
 
for col in ['State_Factor','building_class','facility_type']:
    print(f'number of column {col} is : ',df[col].nunique())
    print(f'name of column {col} is : ',df[col].unique())
#intialization of encoders
sf=OneHotEncoder()#building class encoded using onehot encoder
bc=LabelEncoder()#state factor encoded using label encoder
fc=ce.BinaryEncoder(cols=["facility_type"])#state factor encoded using binary encoder
df['building_class'] = bc.fit_transform(df['building_class'])
facility_encoded_df = fc.fit_transform(df[['facility_type']])

new_encoded = sf.fit_transform(df[['State_Factor']]).toarray()  # Convert to dense array
new_encoded_df = pd.DataFrame(new_encoded, columns=sf.get_feature_names_out(['State_Factor']))
# Drop the original 'State_Factor' column and concatenate the encoded columns
df = pd.concat([df.drop("State_Factor", axis=1), new_encoded_df], axis=1)
df = pd.concat([df.drop("facility_type", axis=1), facility_encoded_df], axis=1)

print(df.info())
print(df.isnull().sum().sort_values(ascending=False))