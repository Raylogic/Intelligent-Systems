import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

cost_list = []

# Calculate the Within-Cluster Sum of Squares (WCSS)
# Args:
#   1) X (list) --> List of examples
#   2) data (list) --> List of examples
#   3) y (list) --> List of predictors of each example
#   4) a (number) --> Learning rate
# Return:
#   1) WCSS

def cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum = sum + np.sqrt((centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
    return sum

#---------------------------------------------------------------------------------------------------------------------------

# Calculate the Within-Cluster Sum of Squares (WCSS)
# Args:
#   1) X (list) --> List of examples
#   2) k (number) --> Number of centroids
# Return:
#   1) Centroids --> Coordinates of the found centroids
#   2) Cluster --> 

def kmeans(X, k):
    diff = 1

    # Generate a list of zeros to assign the centroid index to each example
    cluster = np.zeros(X.shape[0])

    # Get K random indices between 1-len(X)
    random_indices = np.random.choice(len(X), size=k, replace=False)

    # Extract the examples that will serve as centroids from the K random indices
    centroids = X[random_indices, :]

    # Perform the K-means clustering until finding the optimal centroids
    while diff:
        # Cluster the data per centroid
        for i, row in enumerate(X):

            # Assume initial distance from the centroid is infinite
            mean_d = float('inf')

            # Calculate the distance of each point to all centroids
            for idx, centroid in enumerate(centroids):

                # Distance to centroid = Euclidean distance
                d = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                
                # Store the closest distance
                if mean_d > d:
                    mean_d = d
                    cluster[i] = idx
        
        # Group the examples by the current calculated centroids
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values

        # If the newer centroids are same to the old ones, finish the K-means
        # If the newer centroids are better to the old ones, continue the K-means
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids

    return centroids, cluster

#---------------------------------------------------------------------------------------------------------------------------

# Calculate the Within-Cluster Sum of Squares (WCSS)
# Args:
#   1) X (list) --> List of examples
# Return:
#   

def WCSS(X):
    # Calculate the cost of clustering the data into K centroids
    for k in range(1,10):
        # Perform the K-Means with K-centroids
        centroids, cluster = kmeans(X, k)

        # Calculate the K-centroids WCSS
        WCSS = cost(X, centroids, cluster)

        # Add current K-centroids WCSS to cost plot
        cost_list.append(WCSS)

    # Plot the WCSS
    sns.lineplot(x=range(1,10), y=cost_list, marker='o')

    # Plot decorations
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

# Calculate the Within-Cluster Sum of Squares (WCSS)
# Args:
#   1) X (list) --> List of examples
#   2) k (number) --> Number of centroids
# Return:
#   

def clustering(X, k, labels):
    # Plot the dataset
    sns.scatterplot(X[:,0], X[:,1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

    # Calculate the centroids
    # Cluster the data with the centroids
    centroids, cluster = kmeans(X, k)
    print(centroids)

    # Plot the data based on its cluster
    sns.scatterplot(X[:,0], X[:, 1], hue=cluster)

    # Plot the centroids
    sns.scatterplot(centroids[:,0], centroids[:, 1], s=100, color='y')

    # Plot decorations
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

# Select the pair of stats to cluster through K-Means
# Args:
#   
# Return:
#   1) List of stats names in dataset
#   2) List of stats names as labels

def menu():
    # Menu
    stat_options = {}
    stat_options['1'] = "HP" 
    stat_options['2'] = "Attack"
    stat_options['3'] = "Defense"
    stat_options['4'] = "Special Attack" 
    stat_options['5'] = "Special Defense"
    stat_options['6'] = "Speed"

    # Stats dataframe columns
    stat_chosen = []

    # Stats plot labels
    stat_names = []

    while True:
        # Return the pair od stats selected
        if len(stat_chosen) == 2:
            return stat_chosen, stat_names

        # Print the menu
        options = stat_options.keys()
        print("List of stats:")
        for entry in options: 
            print('{}) {}'.format(entry, stat_options[entry]))

        # Detect whether you are selecting the X or Y axis
        if len(stat_chosen) == 0:
            stat = input('Select the 1° stat (X-axis) = ')
        if len(stat_chosen) == 1:
            stat = input('Select the 2° stat (Y-axis) = ')

        # Detect the stat selected by the user
        # Insert the stat name into the dataframe column list
        # Insert the stat name into the plot labels list
        if stat in ['1', 'HP', 'hp']:
            stat_chosen.append('hp')
            stat_names.append('HP')

        elif stat in ['2', 'Attack', 'attack', 'Atk']:
            stat_chosen.append('attack')
            stat_names.append('Attack')

        elif stat in ['3', 'Defense', 'defense']:
            stat_chosen.append('defense')
            stat_names.append('Defense')

        elif stat in ['4', 'Special Attack', 'special attack']:
            stat_chosen.append('sp_attack')
            stat_names.append('Special Attack')

        elif stat in ['5', 'Special Defense', 'special defense']:
            stat_chosen.append('sp_defense')
            stat_names.append('Special Defense')

        elif stat in ['6', 'Speed', 'speed']:
            stat_chosen.append('speed')
            stat_names.append('Speed')

        else:
            print('Please select a valid option')

#---------------------------------------------------------------------------------------------------------------------------

# Perform K-Means clustering on Pokemon Stats
# Args:
#   
# Return:
#   1) WCSS plot
#   2) K-Means cluster plot

def main():

    # Load dataset
    data = pd.read_csv('Pokemon.csv')
    data.head()
    
    # Select the pair of stats to cluster
    stats, labels = menu()
    print(stats)

    # Extract the pair of stats values
    data = data.loc[:, stats]
    X = data.values

    # Plot the WCSS
    WCSS(X)
    
    # Cluster the data into K centroids
    k = int(input("Number of centroids = "))
    clustering(X, k, labels)

main()