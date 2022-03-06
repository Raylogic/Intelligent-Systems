from inspect import Attribute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_errors = []

# Evaluate hypothesis with current parameters
# Args:
#   1) m (list) --> List of parameters of each x
#   2) data (list) --> List of examples
# Return:
#   1) h(x) (number) --> Evaluation of hypothesis

def hyp(m, data):
    h = 0
    # Update h with each calculated parameter
    # h = m1x1 + m2x2 + ... + mnxn
    for i in range(len(data)):
        h = h + m[i] * data[i]

    # Return the hypothesis value
    return h

# ---------------------------------------------------------------------------------------------------------------

# Calculate the MSE of an epoch
# MSE = 1/n * sum((yhi - yi) ^ 2)
# Args:
#   1) m (list) --> List of parameters of each x
#   2) data (list) --> List of examples
#   3) y (list) --> List of predictors of each example
# Return:
#   1) MSE (number) --> Mean Square Error

def MSE(m, data, y):
    acum = 0
    # Calculate the summation
    for i in range(len(data)):

        # Calculate yhi
        yh = hyp(m, data[i])

        # Calculate sum((yhi - yi) ^ 2)
        acum = acum + ((yh - y[i]) ** 2)

    # Calculate MSE (1/n * summation)
    MSE = acum / len(data)

    # Add current epoch MSE to GD error plot
    plot_errors.append(MSE)

# ---------------------------------------------------------------------------------------------------------------

# Use GD to calculate the new values of the parameters
# m = m - a/n * sum((yhi - yi) * xi)
# Args:
#   1) m (list) --> List of parameters of each x
#   2) data (list) --> List of examples
#   3) y (list) --> List of predictors of each example
#   4) a (number) --> Learning rate
# Return:
#   1) new_m (list) --> List of updated parameters of each x

def GD(m, data, y, a):
    # List for updated parameters
    new_m = list(m)

    # Calculate each parameter
    for i in range(len(m)):
        acum = 0
        error = 0

        # Calculate the summation
        for j in range(len(data)):

            # Calculate yhi
            yh = hyp(m, data[j])

            # Calculate yhi - yi
            error = yh - y[j]

            # Calculate (yhi - yi) * xi
            acum = acum + error * data[j][i]

        # Update the parameter
        new_m[i] = m[i] - a/len(data)*acum

    # Return the list of updated parameters
    return new_m

# ---------------------------------------------------------------------------------------------------------------

# Scale examples attributes thorugh mix-max normalization
# Args:
#   1) data (list) --> List of examples
# Return:
#   1) data (list) --> List of examples scaled

def scaling(data):
    acum = 0

    # Transpose the dataset
    data = np.asarray(data).T.tolist()

    # Calculate the scaled attributes of each example
    # Scale = (Value - Mean)/(Max - Min)
    # It is assumed that Min = 0
    for i in range(1, len(data)):

        # Calculate the sum of all attributes values
        for j in range(len(data[i])):
            acum = acum + data[i][j]

        # Calculate the mean of the attributes values
        mean = acum / len(data[i])

        # Calculate the max value of the attributes values
        maxval = max(data[i])

        # Scale each attribute value with the formula
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - mean) / maxval

    # Return the scaled dataset to its original form
    return np.asarray(data).T.tolist()

# ---------------------------------------------------------------------------------------------------------------

# Add bias to the examples
# Args:
#   1) data (list) --> List of examples
# Return:
#   1) data (list) --> List of examples with bias

def bias(data):
    # Add bias to each example
    for i in range(len(data)):

        # Add the bias if the example is already a list
        # Create a list with the bias if the example is not a list
        if isinstance(data[i], list):
            data[i] = [1] + data[i]
        else:
            data[i] = [1,data[i]]

    # Return the examples with bias
    return data

# ---------------------------------------------------------------------------------------------------------------

# Get the Linear Regression of a dataset using Gradient Descent
# Args:
#   1) m (list) --> List of parameters of each x
#   2) data (list) --> List of examples
#   3) y (list) --> List of predictors of each example
#   4) a (number) --> Learning rate
# Return:
#   1) MSE plot
#   2) List of final parameters of each x

def LinearRegression(m, data, y, a):
    # Declare the number of epochs to calculate
    limit = int(input("Epoch limit = "))
    epochs = 0
    while True:
        # Current epoch values 
        # print("Epoch #{}".format(epochs))
        # print("Old parameters = {}".format(m))

        # Calculate the new parameters and MSE
        old_m = list(m)
        m = GD(m, data, y, a)
        MSE(m, data, y)

        # New epoch values 
        # print("New parameters = {}".format(m))
        epochs = epochs + 1

        # End if the parameters don't change between epochs or the epoch limit is reached
        if old_m == m or epochs == limit:
            # print("Data = {}".format(data))
            print("Parameters = {}".format(m))
            break

    # Plot the MSE graph
    plt.plot(plot_errors)
    plt.show()

# ---------------------------------------------------------------------------------------------------------------

# Preprocess the data to use it on a hand-made Linear Regression
# Args:
#   1) df_x (list) --> List of attributes
#   2) df_y (list) --> List of predictors
# Return:
#   1) Linear Regression of the data

def HandLinearRegression(df_x, df_y):
    # Transform the attributes and predictors from pandas dataframes into lists
    a = 0.01
    m = [0,0,0,0]
    data = pd.DataFrame(df_x).to_numpy().tolist()
    list_y = pd.DataFrame(df_y).to_numpy().tolist()

    # Flat the predictors list of list
    y = []
    for i in range(len(list_y)):
        for j in list_y[i]:
            y.append(j)

    data = bias(data)
    data = scaling(data)
    LinearRegression(m, data, y, a)

# ---------------------------------------------------------------------------------------------------------------

# Preprocess the data to use it on a sci-kit Polynomial Regression with linspace testing
# Args:
#   1) df_x (list) --> List of attributes
#   2) df_y (list) --> List of predictors
# Return:
#   1) Polynomial Regression of the data
#   2) Plot of each parameter

def PolynomialRegressionPrediction(degree, df_xi, df_y, attribute):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Declare the polynomial regression paramteres
    poly = PolynomialFeatures(degree, include_bias=False)

    # Transform the attributes to fit the polynomial form
    poly_features = poly.fit_transform(df_xi)

    # Generate the training and test sets
    x_train, x_test, y_train, y_test = train_test_split(poly_features, df_y, test_size=0.3, random_state=42)

    # Declare the polynomial regression
    poly_model = LinearRegression()

    # Perform the Polynomial Regression
    poly_model.fit(x_train, y_train)

    # Sort the attributes from the test set
    x_test = sorted(x_test, key = lambda x: x[0])

    # Predict the y-values of the test set
    poly_prediction = poly_model.predict(x_test)

    # Calculate the MSE
    poly_MSE = np.sqrt(mean_squared_error(y_test, poly_prediction))

    # Print the MSE and regression final parameters
    print(poly_MSE)
    print(poly_model.coef_)

    # Flat the predictors list of list of the test set
    Y_pred = []
    for i in range(len(poly_prediction)):
        for j in poly_prediction[i]:
            Y_pred.append(j)

    # Flat the attributes list of list of the test set
    x_test_graph = []
    for i in range(len(x_test)):
        x_test_graph.append(x_test[i][0])

    # Plot the polynomial regression
    plt.figure(figsize=(12,7))

    # Plot the training set examples
    plt.scatter(x_train[:,0], y_train)

    # Plot the line of the test set examples
    plt.plot(x_test_graph, Y_pred, color='blue', linewidth = 2)
    
    # Plot decorations
    plt.xlabel(attribute)
    plt.ylabel("Comments")
    plt.show()

# ---------------------------------------------------------------------------------------------------------------

# Preprocess the data to use it on a sci-kit Polynomial Regression with linspace testing
# Args:
#   1) df_x (list) --> List of attributes
#   2) df_y (list) --> List of predictors
# Return:
#   1) Polynomial Regression of the data

def ScikitPolynomialRegression(df_x, df_y):
    degree = int(input("Degree = "))

    # Polynomial regression Seconds-Comments
    df_x1 = df_x[["seconds"]]
    PolynomialRegressionPrediction(degree, df_x1, df_y, "Seconds")

    # Polynomial regression Views-Comments
    df_x2 = df_x[["views"]]
    PolynomialRegressionPrediction(degree, df_x2, df_y, "Views")

    # Polynomial regression Likes-Comments
    df_x3 = df_x[["likes"]]
    PolynomialRegressionPrediction(degree, df_x3, df_y, "Likes")

# ---------------------------------------------------------------------------------------------------------------

# Perform Regression on Youtube Statistics
def main():
    # Load dataset
    columns = ["id", "seconds", "views", "likes", "comments"]
    df = pd.read_csv('MrBeast.csv')

    # Define attributes and predictors
    df_x = df[["seconds","views","likes"]]
    df_y = df[["comments"]]

    # Regressions
    HandLinearRegression(df_x, df_y)
    ScikitPolynomialRegression(df_x, df_y)

# MAIN 
main()

# Bibliography:
# 1) Shi, A. (2020), "Polynomial Regression with Scikit learn: What You Should Know" (06/03/2022), Recovered from: https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2
# 2) Ujhelyi, T. (2021), "Polynomial Regression in Python using scikit-learn (with a practical example)" (06/03/2022), Recovered from: https://data36.com/polynomial-regression-python-scikit-learn/

