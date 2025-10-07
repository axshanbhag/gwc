import pandas as pd
import matplotlib.pyplot as plt
# full dataframe info
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

movieData = pd.read_csv('./rotten_tomatoes_movies.csv')
favMovie = "Percy Jackson & the Olympians: The Lightning Thief"
print("My favorite movie is " + favMovie)



#first five rows of movieData
print(movieData.head())
# all titles in the movie_title column
print(movieData["movie_title"])


#Part 4 Filter data
print("\nThe data for my favorite movie is:\n")
#finds column of the movie title
favMovieBooleanList = movieData["movie_title"] == favMovie
#print(favMovieBooleanList)
# creates a new dataframe with rows from the list, or in this case the movie
favMovieData = movieData.loc[favMovieBooleanList]
print(favMovieData)


print("\n\n")

#Create a new variable to store a new data set with a certain genre
dramaMovieBooleanList = movieData["genres"].str.contains("Drama")
# creates a new dataframe with rows with the category of drama
dramaMovieData = movieData.loc[dramaMovieBooleanList]

# number of rows in the dramaMovie.
# shape = (rows, columns)
numOfMovies = dramaMovieData.shape[0]

print("We will be comparing " + favMovie +
      " to other movies under the genre Drama in the data set.\n")
print("There are " + str(numOfMovies) + " movies under the category Drama.")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
input("Press enter to see more information about how " + favMovie +
      " compares to other movies in this genre.\n")

#Part 5 Describe data
#min
min = dramaMovieData["audience_rating"].min()
print("The min audience rating of the data set is: " + str(min))
print(favMovie + " is rated " +str(53-min)+ " points higher than the lowest rated movie.")
print()

#find max
max = dramaMovieData["audience_rating"].max()
print("The max audience rating of the data set is: " + str(max))
print(favMovie + " is rated " +str(max-53)+ " points lower than the highest rated movie.")
print()

#find mean
mean = dramaMovieData["audience_rating"].mean()
print("The mean audience rating of the data set is: " + str(mean))
print(favMovie + " lower than the mean movie rating.")

#find median
median = dramaMovieData["audience_rating"].median()
print("The median audience rating of the data set is: " + str(median))
print(favMovie + " lower than the median movie rating.")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
input("Press enter to see data visualizations.\n")

#Part 6 Create graphs
#Create histogram
plt.hist(dramaMovieData["audience_rating"], range = (0,100), bins = 20)

#Adds labels and adjusts histogram
plt.grid(True)
plt.title("Audience Ratings of Drama Movies Histogram")
plt.xlabel("Audience Ratings")
plt.ylabel("Number of Drama Movies")

#Prints interpretation of histogram
print(
  "According to the histogram, most movies have ratings from 75 to 90. There are about 900 movies with these ratings. There is a significant difference between the ratings from 85 to 90, 90 to 95, and 95 to 100 due to the large slope between the bins."
)
print("Close the graph by pressing the 'X' in the top right corner.")
print()

#Show histogram
plt.show()

#Create scatterplot
plt.scatter(data = dramaMovieData, x = "audience_rating", y = "critic_rating")

#Adds labels and adjusts scatterplot
plt.grid(True)
plt.title("Audience Rating vs Critic Rating")
plt.xlabel("Audience Rating")
plt.ylabel("Critic Rating")
plt.xlim(0, 100)
plt.ylim(0, 100)

#Prints interpretation of scatterplot
print(
  "According to the scatter plot, there is a positive correlation between the Audience Ratings and Critic Ratings."
)
print()

print("Close the graph by pressing the 'X' in the top right corner.")

#Show scatterplot
plt.show()

print("\nThank you for reading through my data analysis!")
