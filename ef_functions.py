import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

#calculate avg compounded yearly returns for each asset and annualize them (now not commpounded)
def calculate_avg_returns(data):
    expected_returns = data.mean()*12
    return expected_returns

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
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
    
    x = efficient_frontier['Standard Deviation (%)']
    y = efficient_frontier['Expected Return (%)']

    # Plot the scatter
    ax.scatter(x, y, marker='o')

    # Annotate each point
    for i in range(len(efficient_frontier)):
        label = f"Portfolio {i+1}"
        ax.annotate(
            label,
            (x.iloc[i], y.iloc[i]),
            textcoords="offset points",
            xytext=(5, 5),  # Offset text position
            ha='left',
            fontsize=8
        )

    # Axis titles and grid
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk (Standard Deviation) (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.grid(True)
    
    return fig

#trying the above but with plotly
def graph2(efficient_frontier):
    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=efficient_frontier['Standard Deviation (%)'],
        y=efficient_frontier['Expected Return (%)'],
        mode='markers+text',
        text=[f"Portfolio {i+1}" for i in range(len(efficient_frontier))],
        textposition="top right",
        marker=dict(size=8),
        name='Portfolios'
    ))
    xmax = max(efficient_frontier['Standard Deviation (%)']) 
    xmin = min(efficient_frontier['Standard Deviation (%)'])
    xrange = xmax - xmin

    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Risk (Standard Deviation) (%)',
        yaxis_title='Expected Return (%)',
        template='plotly_white',
        height=600,
        width=800,
        xaxis=dict(range=[xmin - 0.05 * xrange, xmax + 0.1 * xmax]),
    )
    return fig


#function to find expected returns for second page
def expected_return(rows, returns):
    returns = returns.tail(rows).mean()*12
    return returns

#function to find expected risk for second page
def st_dev(rows, returns):
    vol = returns.tail(rows).std()
    vol_annualized = vol * 12**0.5
    return vol_annualized

#function to make table for second page

def make_table(s1, s2, s3, s4, s5, s6, s7, s8):
    historical = pd.DataFrame({
        s1.name: s1,
        s2.name: s2,
        s3.name: s3,
        s4.name: s4,
        s5.name: s5,
        s6.name: s6,
        s7.name: s7,
        s8.name: s8
    }) * 100

    # Ensure numeric and round
    historical = historical.apply(pd.to_numeric, errors="coerce").round(2)
    historical = historical.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    # Style the DataFrame
    styled = historical.style.set_properties(**{
        'text-align': 'center'
    }).set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]}
    ])

    return styled


#function to graph for second page
def historical_graph(asset_df):
    fig = px.line(asset_df, x="Month", y=["S&P 500", "Fixed Income", "Gold", "Private Credit", "Real Estate", "Private Equity"], 
              title="Historical Asset Prices",
              labels={"value": "Price", "variable": "Asset"},
              markers=True)
    return fig