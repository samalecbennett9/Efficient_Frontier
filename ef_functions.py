import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file):
    return pd.read_csv(file)

def rename(assets):
    selected_assets = [asset + ' Weight' for asset in assets]
    return selected_assets

def monthly_return(data):
    #Add "monthly return" to all columns besides Month
    data.columns = [col + " Monthly Return" if col != "Month" else col for col in data.columns]

    # Make every value a pct change except for the 'Month' column
    data.loc[:, data.columns != "Month"] = data.loc[:, data.columns != "Month"].pct_change()
    asset_data_returns = data.dropna()
    asset_data_returns = asset_data_returns.loc[:, asset_data_returns.columns.str.contains("Return")]
    return asset_data_returns

#calculate avg compounded yearly returns for each asset and annualize them
def calculate_avg_returns(data):
    expected_returns = data.mean()
    expected_returns_compounded = (1 + expected_returns)**12 - 1
    return expected_returns_compounded

# Optimization
def optimize(target_returns, weights, cov_matrix, expected_returns):
    efficient_portfolios = []
    for target_return in target_returns:
        # Objective: Minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        # Constraints
        constraints = [
            weights >= 0, # No short-selling
            cp.sum(weights) == 1, # Fully invested
            expected_returns.values @ weights >= target_return # Target return
            ]
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        if prob.status == 'optimal':
            annualized_return = expected_returns.values @ weights.value
            annualized_variance = portfolio_variance.value
            efficient_portfolios.append({
                'Weights': weights.value,
                'Return': annualized_return,
                'Variance': annualized_variance
                })
    return efficient_portfolios

def clean_and_table(efficient_frontier, selected_assets):
    # Split the 'Weights' into independent columns
    efficient_frontier[selected_assets] = pd.DataFrame(efficient_frontier['Weights'].tolist(), index=efficient_frontier.index)

    # Convert weights to percentages
    efficient_frontier = efficient_frontier * 100

    # Add '%' to the asset column names
    efficient_frontier.columns = [f"{col} (%)" if 'Weight' in col else col for col in efficient_frontier.columns]

    # Round the values to 2 decimal places
    efficient_frontier = efficient_frontier.round(2)

    # Change the index to 'Portfolio X' format
    efficient_frontier.index = [f"Portfolio {i+1}" for i in range(len(efficient_frontier))]

    # Rename columns
    efficient_frontier = efficient_frontier.rename(columns={'Return': 'Expected Return (%)', 
                                                        'Variance': 'Variance (%)', 
                                                        'Standard Deviation': 'Standard Deviation (%)'})
    #Drop Weights columns
    efficient_frontier = efficient_frontier.drop("Weights (%)", axis=1)

    #Drop Var Column
    efficient_frontier = efficient_frontier.drop("Variance (%)", axis=1)
    return efficient_frontier

#Graph
def graph(efficient_frontier):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        efficient_frontier['Standard Deviation (%)'],
        efficient_frontier['Expected Return (%)'],
        marker='o'
    )
    plt.title('Efficient Frontier')
    plt.xlabel('Risk (Standard Deviation) (%)')
    plt.ylabel('Expected Return (%)')
    plt.grid(True)
    plt.show()