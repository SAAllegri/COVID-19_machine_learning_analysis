import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
# df = pd.read_csv("data/complete_joined_data.csv") # change filepath if needed
df = pd.read_csv("/Users/Sarah/Desktop/GT/Spring 2021/CS 7641/Project/Data/cleaned_data_2/cases_deaths_vaccines_cleaned.csv")

df_filtered = df.dropna(axis=0) # drop any rows with nan or missing values
df_filtered = df_filtered.drop(['day_count', 'submission_date'], axis=1) # drop these columns- I think we don't need the data in them

# Not sure what to do with these 2 columns because I feel like they're important information, but the dates are stored as strings
# And we can't pass strings to PCA- they have to be floats
# Not sure how is the best way to convert string date to float date and have it still be meaningful
# Just dropping the entire columns for now until we figure out what to do with them
# df_filtered = df_filtered.drop(['date_stay_at_home_announced', 'date_stay_at_home_effective'], axis=1)

# Separate out the target
y = df_filtered.loc[:,['state_x', 'state_y', 'total_vaccinations', 'date']].values # y is target values/labels- the number of vaccinations for each state

# Separate out the features
headers = list(df_filtered.columns)# get all the column names
headers.remove("state_x") # exclude these two columns because they are the targets
headers.remove("state_y")
headers.remove("total_vaccinations")
headers.remove("date")
x = df_filtered.loc[:, headers].values # x is the datapoints

# Standardize the features
x = StandardScaler().fit_transform(x)

# Do PCA on the dataset
# We can change the value of n_components. 20 was just the default
# We probably need a good way to figure out the best value of n_components though
# Create PCA object, determine how many features we want to retain (n_components)
# pca = PCA(n_components=3) # do PCA with 3 components
pca = PCA() # do PCA with all components
reconstructX = pca.fit_transform(x) # reconstruct data using principal components
principalComponents = pca.components_
pc_variance = pca.explained_variance_ratio_ # the percentage of total variance that is contained within each feature
# print(pca.explained_variance_ratio_.cumsum()) # total amount of variance contained within n_ features

# Plot scree plot
pc_number = np.arange(len(pc_variance)) + 1
plt.plot(pc_number, pc_variance, "-o")
plt.title('Cases/Vaccines/Deaths Scree Plot')
plt.xlabel("Principal Component")
plt.ylabel("Variance")

# Reconstruct data
reconstructDf = pd.DataFrame(data = reconstructX) # data reconstructed using only reduced features
finalDf = pd.concat([reconstructDf, df_filtered[['state_x', 'state_y', 'total_vaccinations', 'date']]], axis = 1) # add the states and total cases values back in
finalDf.to_csv(r'/Users/Sarah/Desktop/GT/Spring 2021/CS 7641/Project/PCA/with_dates/cases_deaths_vaccines_3_components_with_dates.csv', index=False)