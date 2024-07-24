#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("C:/Users/MSI/PycharmProjects/pythonProject/SALES/Oil and Gas 1932-2014.csv")

# List of countries in the dataset
countries = data['cty_name'].unique()

# Calculate global annual oil production
annual_global_oil_prod = data.groupby('year')['oil_prod32_14'].sum()

# Function to plot annual oil production for a given country compared to global production
def plot_country_oil_production(country):
    figgg, ax1 = plt.subplots(figsize=(14, 7))  # Create a new figure for each plot
    country_data = data[data['cty_name'] == country]
    annual_country_oil_prod = country_data.groupby('year')['oil_prod32_14'].sum()

    # Plot the country's oil production trend
    ax1.fill_between(annual_country_oil_prod.index, annual_country_oil_prod.values, color='#ff7f0e', alpha=0.5,
                     label=f'{country}')
    ax1.set_xlabel('Year')
    ax1.set_ylabel(f'{country} Oil Production (million barrels per day)', color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#ff7f0e')

    ax1.fill_between(annual_global_oil_prod.index, annual_global_oil_prod.values, color='#1f77b4', alpha=0.3,
                     label='Global')
    ax1.set_ylabel('Global Oil Production (million barrels per day)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Title and legend
    plt.title(f'Oil Production Trend for {country} Compared to Global (1932-2014)')
    figgg.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
    plt.show()

# Create the Dropdown widget
country = widgets.Dropdown(options=list(countries))

# Link the dropdown with the plotting function using `interact`
widgets.interact(plot_country_oil_production, country=country)


# In[29]:


# Create a DataFrame for the annual oil production
oil_prod_df = pd.DataFrame({'year': annual_global_oil_prod.index, 'oil_prod': annual_global_oil_prod.values})

# Calculate the historical mean of oil production
historical_mean = oil_prod_df['oil_prod'].mean()

# Define a mean reversion model
def mean_reversion_model(current_value, mean, reversion_speed=0.1, years=10):
    predictions = []
    for _ in range(years):
        current_value += reversion_speed * (mean - current_value)
        predictions.append(current_value)
    return predictions

# Fit a linear regression model
X = oil_prod_df[['year']]
y = oil_prod_df['oil_prod']
linear_model = LinearRegression()
linear_model.fit(X, y)

# Fit a polynomial regression model (degree 5 for this example) using np.polyfit
degree = 5
poly_coeffs = np.polyfit(oil_prod_df['year'], oil_prod_df['oil_prod'], degree)
poly_model = np.poly1d(poly_coeffs)

# Create future years from 1932 to 2024
future_years = np.arange(1932, 2025)
future_years_poly = np.polyval(poly_coeffs, future_years)

# Predict future oil production using linear regression
linear_predictions = linear_model.predict(future_years.reshape(-1, 1))

# Predict future oil production using mean reversion
current_value = oil_prod_df['oil_prod'].iloc[-1]
mean_reversion_predictions = mean_reversion_model(current_value, historical_mean, reversion_speed=0.1, years=len(future_years))

# Plot the historical data and the predictions
plt.figure(figsize=(14, 7))

# Historical data
sns.lineplot(x=oil_prod_df['year'], y=oil_prod_df['oil_prod'], label='Historical Data')

# Linear regression predictions
plt.plot(future_years, linear_predictions, label='Linear Regression Predictions', linestyle='--')

# Polynomial regression predictions
plt.plot(future_years, future_years_poly, label='Polynomial Regression Predictions (5th degree)', linestyle='--')

# Mean reversion predictions
plt.plot(future_years, mean_reversion_predictions, label='Mean Reversion Predictions', linestyle='--')

plt.title('Global Oil Production Trend and Predictions (1932-2024)')
plt.xlabel('Year')
plt.ylabel('Oil Production (million barrels per day)')
plt.grid(True)
plt.legend()
plt.show()



# In[31]:


# Extracting relevant columns
oil_prices = data[['year', 'oil_price_nom', 'oil_price_2000']]

# Plot the trends
plt.figure(figsize=(14, 7))
sns.lineplot(x=oil_prices['year'], y=oil_prices['oil_price_nom'], label='Nominal Price')
sns.lineplot(x=oil_prices['year'], y=oil_prices['oil_price_2000'], label='Real Price (2000)')
plt.title('Oil Prices Trend (1932-2014)')
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[47]:


# Filter out only relevant columns
relevant_data = data[['cty_name', 'year', 'oil_prod32_14']]

# Initial year for the plot
initial_year = 2014

# List of colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Function to plot top oil producing countries for a given year
def plot_top_oil_producers(year):
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    oil_prod_year = relevant_data[relevant_data['year'] == year]
    top_oil_producers_year = oil_prod_year.sort_values(by='oil_prod32_14', ascending=False).head(10)

    ax.bar(top_oil_producers_year['cty_name'], top_oil_producers_year['oil_prod32_14'], color=colors)
    ax.set_title(f'Top Oil Producing Countries in {year}')
    ax.set_xlabel('Country')
    ax.set_ylabel('Oil Production (million barrels per day)')
    ax.set_xticks(top_oil_producers_year['cty_name'])
    ax.set_xticklabels(top_oil_producers_year['cty_name'], rotation=45)
    plt.show()

# Display the interactive widget with a plot
widgets.interact(plot_top_oil_producers, year=(1932, 2014, 1))


# In[37]:


# Historical events
events = {
    1973: '1973 Oil Crisis',
    1979: '1979 Oil Crisis',
    1990: 'Gulf War',
    2008: 'Global Financial Crisis'
}

# Plot oil prices with events
plt.figure(figsize=(14, 7))
sns.lineplot(x=oil_prices['year'], y=oil_prices['oil_price_2000'], label='Real Price (2000)')
for year, event in events.items():
    plt.axvline(x=year, color='r', linestyle='--')
    plt.text(year+1, oil_prices[oil_prices['year'] == year]['oil_price_2000'].values[0] + 5, event, rotation=45)
plt.title('Impact of Historical Events on Oil Prices (1932-2014)')
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




