import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocess the data to use it on a sci-kit Polynomial Regression with linspace testing
# Args:
#   1) degree --> Degree of the regression
#   2) df_x (list) --> List of attributes
#   3) df_y (list) --> List of predictors
# Return:
#   1) Polynomial Regression of the data
#   2) Plot of the regression

def SinglePolynomialRegression(df_x, df_y, degree, xlabel):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Declare the polynomial regression paramters
    poly = PolynomialFeatures(degree, include_bias=False)

    # Transform the attributes to fit the polynomial form
    poly_features = poly.fit_transform(df_x)

    # Generate the training and test sets
    x_train, x_test, y_train, y_test = train_test_split(poly_features, df_y, test_size=0.2, random_state=42)

    # Declare the polynomial regression
    poly_model = LinearRegression()

    # Perform the Polynomial Regression
    poly_model.fit(x_train, y_train)

    # Predict the values of the test set
    y_pred = poly_model.predict(x_test)

    # Calculate the MSE
    MSE = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate the coefficient of determination
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print("Parameters = {}".format(poly_model.coef_))
    print("MSE = {}".format(MSE))
    print("R2 = {}".format(r2))

    # Insert test set paramenters and predicitons into lists
    x_test_val = [item[0] for item in x_test]
    y_pred_val = [item[0] for item in y_pred]

    # Append test set paramenters and predicitons into the real values Dataframe
    y_test['x_test'] = x_test_val
    y_test['y_pred'] = y_pred_val

    # Sort the dataframe by the test set parameters
    y_test.sort_values(by=['x_test'])

    # Plot the test examples
    # Plot the predicitons
    plt.figure(figsize=(10,6))
    plt.scatter(y_test['x_test'], y_test['hp'])
    plt.scatter(y_test['x_test'], y_test['y_pred'], color='#9200F3')
    plt.xlabel(xlabel)
    plt.ylabel('HP')
    plt.show()
   
# ---------------------------------------------------------------------------------------------------------------

# Perform Regression on Youtube Statistics
def main():
    # Load dataset
    columns = ["name", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    df = pd.read_csv('Pokemon.csv')

    # Define attributes
    stats = ["attack", "defense", "sp_attack", "sp_defense", "speed"]
    names = ["Attack", "Defense", "Special Attack", "Special Defense", "Speed"]

    # Define predictors
    df_y = df[["hp"]]

    # Declare the degreee of the polynomial regression
    degree = int(input("Degree = "))

    # Perform the Polynomial Regression with each attribute
    for i in range(len(stats)):
        # Define the current attribute
        df_x = df[[stats[i]]]

        # Do the Single Polynomial Regression
        SinglePolynomialRegression(df_x, df_y, degree, names[i])

# MAIN 
main()
