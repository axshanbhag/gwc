# The pandas library is used for data processing and to read data files
import pandas as pd 
#The matplotlib library is used to plot histograms and scatter plots
import matplotlib.pyplot as plt
# The GWCutilities has functions to help format data printed to the console
import GWCutilities as util

# Read a comma separated values (CSV) files into a variable
# as a pandas DataFrame
lwd=pd.read_csv("livwell135.csv")

# Print out the number of rows and columns in a tuple
print(lwd.shape)
print("\nApproximately 89.32 million females live in Bangladesh, as of 2025, but less than half of them are involved in the labor work force. Oftentimes, women are involved in agricultural work, and bringing back drinking water for their family, as their main job. However, they can be prevented from this opportunty due to poor climate and global warming. Women work in a patriarchal society where they are expected to fulfill household responsibilities.\n")
input("\n Press enter to continue: \n")
print("\nWomen in Bangaldesh have drastic pay check differences and face higher unemployment rates than men because they find it difficult to secure a full-time job. They struggle to receive higher education, and can be forced to have an early marriage. However, Bangladesh is working to combot growing poverty with an increase in education over the past years. Along with their outside work, the boom of the garment industry created the need for an increase in female labor workers, despite a basic level of education. \n")
input("\n Press enter to continue: \n")
print("\nAs denoted by the Women's Well-being dataset's scatterplot for Bangladesh, the number of women working has been fluctuating from 2000 to 2015. It has experienced periods of growth, decline, and stagnation. As of post 2011, we can see a positive increase in the number of women who are legally employed.\n ")
#  basic colors:
# 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'
# find the columnn on Bangladesh
oneCountryBooleanList = lwd["country_name"]=="Bangladesh"
# creates a new dataframe with the row of information on Bangladesh
oneCountryData = lwd.loc[oneCountryBooleanList]
# create a scatter plot of Bangladesh year (x) vs women working (y)
plt.scatter(oneCountryData["year"],oneCountryData["WK_working_p"],color="red")

# add a title to the plot
plt.title("Percent of Women Who are Working over Time")

#Label the x-axis
plt.xlabel("Year")

# label the y-axis
plt.ylabel("Women currently working (%)")

# set the range for the y-axis
plt.ylim(0,oneCountryData["WK_working_p"].max())




# show the plot
plt.show()
