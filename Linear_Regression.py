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
    return MSE

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
        new_m[i] = m[i] - a / len(data) * acum

    # Return the list of updated parameters
    return new_m

# ---------------------------------------------------------------------------------------------------------------

# Calculate the mean of an attribute
# Args:
#   1) column (list) --> List of attributes
# Return:
#   1) Mean

def mean(data):
    return sum(data)/len(data)

# Calculate the standard deviation of an attribute
# Args:
#   1) column (list) --> List of attributes
# Return:
#   1) Standard Deviation

def deviation(data):
    return np.std(data)

# ---------------------------------------------------------------------------------------------------------------

# Scale examples attributes thorugh Standarization
# Args:
#   1) data (list) --> List of examples
# Return:
#   1) data (list) --> List of scaled examples

def scaling(data):
    # Transpose the dataset
    data = np.asarray(data).T.tolist()

    # Calculate the scaled attributes of each example
    for i in range(1, len(data)):

        # Calculate the sum of all attributes values
        for j in range(len(data[i])):

            # Get the values of a dataset column
            x_actual = [item[j] for item in data]

            # Calculate the mean of the column
            x_mean = mean(x_actual)

            # Calculate the standard deviation of the column
            x_std = deviation(x_actual)

            # Perform the Standarization Scaling
            data[i][j] = (data[i][j] - x_mean) / x_std

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
        error = MSE(m, data, y)

        # New epoch values 
        # print("New parameters = {}".format(m))
        epochs = epochs + 1

        # End if the parameters don't change between epochs or the epoch limit is reached
        if old_m == m or epochs == limit:
            # print("Data = {}".format(data))
            print("Parameters = {}".format(m))
            print("MSE = {}".format(error))
            break

    # Plot the MSE graph
    plt.plot(plot_errors)
    plt.show()
    return m

# ---------------------------------------------------------------------------------------------------------------

# Predict the values of the test examples
# Args:
#   1) m (list) --> List of parameters of each x
#   2) x_test (list) --> List of examples
# Return:
#   1) y_pred (list) --> Prediction of examples

def predict(m, data):
    # List of predicted values
    y_pred = []
    
    # Evaluate the test examples with the optimal paramaters
    # Append the hypothesis result into the prediction list
    for i in range(len(data)):
        y = hyp(m, data[i])
        y_pred.append(y)

    # Return the hypothesis value
    return y_pred

# ---------------------------------------------------------------------------------------------------------------

# Calculate the coefficient of determination of the model
# Args:
#   1) y_test (list) --> List of test examples
#   2) y_pred (list) --> List of predicted examples
# Return:
#   1) R2 (number) --> Coefficient of determination

def R2Coefficient(y_test, y_pred):
    # Calculate the mean of y
    y_mean = mean(y_test)

    # Calculate the Sum of Square Residuals (SSR)
    SSR = 0
    for i in range(len(y_test)):
        SSR = SSR + (y_test[i] - y_mean) ** 2
    
    # Calculate the Sum of Square Totals (SST)
    SST = 0
    for i in range(len(y_test)):
        SST = SST + (y_test[i] - y_pred[i]) ** 2

    # Calculate the Coefficient of Determination
    R2 = 1 - (SSR / SST)
    print("R2 = {}".format(R2))

    # Return the Coefficient of Determination
    return R2

# ---------------------------------------------------------------------------------------------------------------

# Preprocess the data to use it on a hand-made Linear Regression
# Args:
#   1) df_x (list) --> List of attributes
#   2) df_y (list) --> List of predictors
# Return:
#   1) Linear Regression of the data

def PreProcessing(df_x, df_y):
    # Transform the attributes and predictors from pandas dataframes into lists
    a = 0.01
    m = [0,0,0,0,0,0]
    data = pd.DataFrame(df_x).to_numpy().tolist()
    list_y = pd.DataFrame(df_y).to_numpy().tolist()

    # Flat the predictors list
    y = []
    for i in range(len(list_y)):
        for j in list_y[i]:
            y.append(j)

    # Split the data into training and test examples
    x_train = data[:-150]
    x_test = data[-150:]
    y_train = y[:-150]
    y_test = y[-150:]

    # Add the bias to the examples
    data = bias(x_train)

    # Scale the examples
    data = scaling(x_train)

    # Perform the Linear Regression
    m = LinearRegression(m, data, y_train, a)  

    # Predict the test examples with the optimal parameters
    y_pred = predict(m, x_test)

    # Calculate the coefficient of determination
    r2 = R2Coefficient(y_test, y_pred)

# ---------------------------------------------------------------------------------------------------------------

# Perform Regression on Pokemon Stats
def main():
    # Load dataset
    columns = ["name", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    df = pd.read_csv('Pokemon.csv')

    # Define attributes and predictors
    df_x = df[["attack", "defense", "sp_attack", "sp_defense", "speed"]]
    df_y = df[["hp"]]

    # Prepare the data for the regression
    PreProcessing(df_x, df_y)

# MAIN 
main()
