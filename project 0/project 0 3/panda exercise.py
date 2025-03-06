import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\vc studio projects\\python files\\assignment 3\\job_data_extended.csv')

print(data.head())

num_rows, num_columns = data.shape
print(f'The dataset has {num_rows} rows and {num_columns} columns.')

print('The average price is:', data['Purchase Price'].mean())
print('The maximum price is:', data['Purchase Price'].max())
print('The minimum price is:', data['Purchase Price'].min())
print('The number of people who speak Arabic is:', data[data['Language'] == 'Arabic'].shape[0])

print('The number of people who are lawyers is:', data[data['Job'].str.lower() == 'lawyer'].shape[0])

count_AM = (data['Purchase Time'] == 'AM').sum()
count_PM = (data['Purchase Time'] == 'PM').sum()

print('The number of AM purchases is:', count_AM)
print('The number of PM purchases is:', count_PM)

print(data['Job'].value_counts().head())

most_common_job = data['Job'].value_counts().idxmax()  
highest_count = data['Job'].value_counts().max() 

print(f'The most common job is: {most_common_job} with {highest_count} occurrences')
