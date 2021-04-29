import pandas as pd
from tqdm import tqdm
import numpy as np


# Converts state abbreviations to full name
def conv_abv(abv):
    idx = abv_dataframe[abv_dataframe['abv'] == abv].index.tolist()
    name = []
    if len(idx) == 1:
        name.append(abv_dataframe.at[idx[0], 'name'])
    else:
        name.append('NA')

    return name[0]


# Splits dataframe into dataframes by state
def split_df_state(df, state_name):
    sub_df = df[df['state'] == state_name]

    return sub_df


# Initialize tqdm for pandas
tqdm.pandas()

# import data (and adjust them) from csv
abv_dataframe = pd.read_csv('../original_data_2/abbr-name.csv', header=None)

cases_deaths_df = pd.read_csv('../original_data_2/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

vaccine_data_df = pd.read_csv('../original_data_2/us_state_vaccinations.csv')

abv_dataframe.columns = ['abv', 'name']

vaccine_data_df = vaccine_data_df.rename(columns={'location': 'state'})

# Convert abbreviations in cases_deaths_df using abv_dataframe
print('Cleaning state abbreviations...')

cases_deaths_df['state'] = cases_deaths_df.progress_apply(lambda x: conv_abv(x['state']), axis=1)

print('Cleaning dataframes and joining...')

# Format date
cases_deaths_df['submission_date'] = pd.to_datetime(cases_deaths_df['submission_date'], format='%m/%d/%Y')
vaccine_data_df['date'] = pd.to_datetime(vaccine_data_df['date'], format='%Y/%m/%d')

# Sort dataframe by date and reset index
cases_deaths_df = cases_deaths_df.sort_values(by='submission_date')

cases_deaths_df = cases_deaths_df.reset_index(drop=True)

# Filter out all but 50 states + DC
cases_deaths_df = cases_deaths_df[cases_deaths_df['state'] != 'NA']

# Filter out undesired columns
cleaned_cases_deaths = cases_deaths_df[
    ['state', 'submission_date', 'tot_cases', 'conf_cases', 'tot_death', 'conf_death']]

# Generate list of state names
state_list = abv_dataframe['name'].to_list()

state_split_dataframes = []
state_split_dataframes_vaccine = []

# Split cleaned_cases_death into list of dataframes for each state
for state in state_list:
    state_split_dataframes.append(split_df_state(cleaned_cases_deaths, state))

# print(state_split_dataframes)

# Split vaccine_data_df into list of dataframes for each state
for state in state_list:
    state_split_dataframes_vaccine.append(split_df_state(vaccine_data_df, state))

# Join cases_deaths and vaccine dataframes
cases_deaths_vaccines_list = []

for i in range(len(state_list)):
    cases_deaths_vaccines_list.append(
        pd.merge(state_split_dataframes_vaccine[i], state_split_dataframes[i], left_on='date', right_on='submission_date', how='inner'))

# List to dataframe
final_df = pd.concat(cases_deaths_vaccines_list)

# Last minute cleaning
final_df.index.names = ['day_count']
final_df[final_df.columns].replace('', np.nan, inplace=True)
final_df.dropna(how='any', inplace=True)

# Put final_df into csv
final_df.to_csv('cases_deaths_vaccines_cleaned.csv')

# print('Done!')
