import pandas as pd
import numpy as np


def stay_at_home_effective_enumerated(row):

    if row['date'] >= row['date_stay_at_home_effective']:

        return 1

    else:

        return 0


def cases_population_proportion(row):

    return row['cases'] / row['total_population']


def deaths_population_proportion(row):

    return row['deaths'] / row['total_population']


print('Reading data from csv...')

# Importing csv files as pandas dataframes and performing a light clean
large_dataframe = pd.read_csv('../original_data_2/US_counties_COVID19_health_weather_data.csv')

abv_dataframe = pd.read_csv('../original_data_2/abbr-name.csv')

abv_dataframe.columns = ['abv', 'name']

state_list = abv_dataframe['name'].to_list()

# Adding DC (formatting problems between state names and large dataframe)
for i in range(len(state_list)):

    if state_list[i] == 'District Of Columbia':

        state_list[i] = 'District of Columbia'


# All the features/columns we want to keep
large_dataframe = large_dataframe[['date', 'state', 'county', 'cases', 'deaths', 'population_density_per_sqmi',
                                   'total_population', 'percent_smokers', 'percent_adults_with_obesity',
                                   'percent_uninsured', 'num_primary_care_physicians',
                                   'percent_vaccinated', 'income_ratio', 'life_expectancy',
                                   'percent_adults_with_diabetes', 'percent_uninsured_2',
                                   'other_primary_care_provider_rate', 'median_household_income',
                                   'percent_less_than_18_years_of_age', 'percent_65_and_over',
                                   'percent_black',
                                   'percent_american_indian_alaska_native', 'percent_asian',
                                   'percent_native_hawaiian_other_pacific_islander', 'percent_hispanic',
                                   'percent_non_hispanic_white', 'percent_not_proficient_in_english',
                                   'percent_female', 'percent_rural', 'per_capita_income',
                                   'percent_below_poverty', 'percent_unemployed_CDC', 'percent_age_65_and_older',
                                   'percent_age_17_and_younger', 'percent_overcrowding', 'date_stay_at_home_effective']]

# Generates .pkl file to store reduced health_weather df
large_dataframe.to_pickle('health_weather_reduced.pkl')

# Removes any row with an empty entry
large_dataframe[large_dataframe.columns].replace('', np.nan, inplace=True)
large_dataframe.dropna(how='any', inplace=True)

large_dataframe = large_dataframe.sort_values(by=['state', 'county', 'date'])

large_dataframe['date_stay_at_home_effective'] = large_dataframe.apply(lambda x: stay_at_home_effective_enumerated(x), axis=1)

large_dataframe['cases_population_proportion'] = large_dataframe.apply(lambda x: cases_population_proportion(x), axis=1)

large_dataframe['deaths_population_proportion'] = large_dataframe.apply(lambda x: deaths_population_proportion(x), axis=1)

# Exports as csv
large_dataframe.to_csv('state_county_info.csv')

print('Done!')