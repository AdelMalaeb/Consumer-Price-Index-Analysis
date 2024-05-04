#!/usr/bin/env python
# coding: utf-8



def evaluate_linear_regression_assumptions(x,y):
    """
    This function automatically evaluates the assumptions of linear regression
    for multiple feautures in a dataset
    
        
    Returns:
        A dictionary containing the results of the check
        
    """
    sns.scatterplot(x=x, y=y, data = data)
    plt.title("Scatter plot")
    plt.show()
    
    #Fit OLS model
   
    ols_formula = "y ~ x"

    #fit the model
    model = ols(formula = ols_formula, data = data)
    model = model.fit()
    print(model.summary())
    
    residuals = model.resid
    
   # Create a figure and subplots grid using Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 rows, 2 column

    # Plot 1: Histogram plot using Seaborn in the first subplot (axs[0])
    sns.histplot(x=residuals, data=data, ax=axs[0])
    axs[0].set_title('Histogram of residuals')

    # Plot 2: QQ plot using Seaborn in the second subplot (axs[1])
    sm.qqplot(residuals, ax=axs[1],line = 's')
    axs[1].set_title('QQ plot of residuals')

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()
    
    fitted_values = model.fittedvalues
    sns.scatterplot(x=fitted_values, y=residuals)

    # Set the x-axis label.
    plt.xlabel("fitted_values")
    # Set the y-axis label.
    plt.ylabel("residuals")
    # Set the title.
    plt.title("Homoscedasticity Assupmtion")
    # Add a line at y = 0 to visualize the variance of residuals above and below 0.
    plt.axhline(y=0, color='red', linestyle='--')
 

    plt.show()


def evaluate_linear_regression_assumptions(x1,x2,y):
    """
    This function automatically evaluates the assumptions of linear regression
    for multiple feautures in a dataset
    
        
    Returns:
        A dictionary containing the results of the check
        
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1 rows, 2 column
    sns.scatterplot(x=x1, y=y, data = data, ax=axes[0])
    sns.scatterplot(x=x2, y=y, data = data, ax=axes[1])
    plt.title("Scatter plot")
    plt.show()
    
    #Fit OLS model
   
    ols_formula = "y ~ x1 + x2"

    #fit the model
    model = ols(formula = ols_formula, data = data)
    model = model.fit()
    print(model.summary())
    
    residuals = model.resid
    
   # Create a figure and subplots grid using Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 rows, 2 column

    # Plot 1: Histogram plot using Seaborn in the first subplot (axs[0])
    sns.histplot(x=residuals, data=data, ax=axs[0])
    axs[0].set_title('Histogram of residuals')

    # Plot 2: QQ plot using Seaborn in the second subplot (axs[1])
    sm.qqplot(residuals, ax=axs[1],line = 's')
    axs[1].set_title('QQ plot of residuals')

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()
    
    fitted_values = model.fittedvalues
    sns.scatterplot(x=fitted_values, y=residuals)

    # Set the x-axis label.
    plt.xlabel("fitted_values")
    # Set the y-axis label.
    plt.ylabel("residuals")
    # Set the title.
    plt.title("Homoscedasticity Assupmtion")
    # Add a line at y = 0 to visualize the variance of residuals above and below 0.
    plt.axhline(y=0, color='red', linestyle='--')
 

    plt.show()





