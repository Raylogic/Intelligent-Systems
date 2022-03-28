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
#   2) Plot of each parameter

def MultiPolynomialRegression(df_x, df_y, degree):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Declare the polynomial regression paramteres
    poly = PolynomialFeatures(degree, include_bias=False)

    # Transform the attributes to fit the polynomial form
    poly_features = poly.fit_transform(df_x)

    # Generate the training and test sets
    x_train, x_test, y_train, y_test = train_test_split(poly_features, df_y, test_size=0.3, random_state=42)

    # Declare the polynomial regression
    poly_model = LinearRegression()

    # Perform the Polynomial Regression
    poly_model.fit(x_train, y_train)

    # Predict the y-values of the test set
    y_pred = poly_model.predict(x_test)

    # Calculate the MSE
    MSE = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate the coefficient of determination
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print("Parameters = {}".format(poly_model.coef_))
    print("MSE = {}".format(MSE))
    print("R2 = {}".format(r2))

# ---------------------------------------------------------------------------------------------------------------

# Perform Regression on Youtube Statistics
def main():
    # Load dataset
    columns = ["name", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    df = pd.read_csv('Pokemon.csv')

    # Define attributes and predictors
    df_x = df[["attack", "defense", "sp_attack", "sp_defense", "speed"]]
    df_y = df[["hp"]]

    # Declare the degreee of the polynomial regression
    degree = int(input("Degree = "))

    # Perform the Multi-Polynomial regression
    MultiPolynomialRegression(df_x, df_y, degree)

# MAIN 
main()
