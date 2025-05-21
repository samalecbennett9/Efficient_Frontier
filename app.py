import ef_functions as ef
import streamlit as st
import cvxpy as cp
import numpy as np
import pandas as pd

#title
st.title("Efficient Frontier")

#info
st.write("This application calculates an Efficient Frontier for up to six assets. The user can choose which assets they would like to incorporate into their portfolio. Options include: S&P 500, Fixed Income, Gold, Private Credit, Real Estate, and Private Equity. The user can also select the time frame of the historical data used in the calculation. All data is ending December 31, 2024.")



#sidebar
st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Efficient Frontier", "Historical Data"])

if page == "Efficient Frontier":

    #Read data
    asset_data = ef.read_data("EF_Data_Summary.csv")

    #make spot date
    spot_date = asset_data['Month'].iloc[-1]

    st.header("Efficient Frontier")
    
    #spot data subheader
    st.subheader(f"The spot date is {spot_date}")

    #user selects assets to include
    assets = ['S&P 500', 'Fixed Income', 'Gold', 'Private Credit', 'Real Estate', 'Private Equity']
    selected_assets = st.multiselect(
        "Choose 2 to 6 assets you would like to include in your Efficient Frontier",
        options=assets,
        default=assets  
    )

    #user selects historical time frame
    time_frame = st.slider("Choose your desired time frame in years (2-10)", 2, 10)

    #Filter data to just include selected assets
    asset_data_filtered = asset_data[selected_assets]

    #Filter data to just include time frame
    included_rows = int(time_frame)*12+1 #changed this (+1)
    asset_data_filtered_twice = asset_data_filtered.tail(included_rows)

    #add "Weight" to selected assets for later use
    selected_assets_with_weight = ef.rename(selected_assets)

    #make the data into monthly returns
    monthly_return_data = ef.monthly_return(asset_data_filtered_twice)

    #make a series with the annualized returns of each asset
    expected_annualized_returns = ef.calculate_avg_returns(monthly_return_data)

    #make the cov matrix
    cov_matrix = asset_data_filtered_twice.cov() * 12

    #Primary Calculations
    n = len(cov_matrix)
    weights = cp.Variable(n)

    #Calculate Target Returns
    target_returns = np.linspace(
    expected_annualized_returns.min(),
    expected_annualized_returns.max(),
    10 # Number of points on the frontier
    )

    #optimize
    efficient_portfolios = ef.optimize(target_returns, weights, cov_matrix, expected_annualized_returns)

    # Convert to DataFrame
    efficient_frontier = pd.DataFrame(efficient_portfolios)
    efficient_frontier['Standard Deviation'] = np.sqrt(efficient_frontier['Variance'])

    # Remove duplicates (if any)
    efficient_frontier = efficient_frontier.drop_duplicates(['Return', 'Variance'])

    #make the portfolio table for first page
    table = ef.clean_and_table(efficient_frontier, selected_assets_with_weight)


    #create the second graph
    graph2 = ef.graph2(table)

    #show table df
    st.dataframe(table)

    #show second graph
    st.plotly_chart(graph2, use_container_width=True)

else:
#create the table for second page
    asset_data_full = ef.read_data("EF_Data_Summary.csv")

    monthly_return_data_full = ef.monthly_return(asset_data_full)

    returns_annualized_10 = ef.expected_return(119, monthly_return_data_full)
    vol_annualized_10 = ef.st_dev(119, monthly_return_data_full)
    returns_annualized_5 = ef.expected_return(60, monthly_return_data_full)
    returns_annualized_2 = ef.expected_return(24, monthly_return_data_full)
    returns_annualized_1 = ef.expected_return(12, monthly_return_data_full)
    vol_annualized_5 = ef.st_dev(60, monthly_return_data_full)
    vol_annualized_2 = ef.st_dev(24, monthly_return_data_full)
    vol_annualized_1 = ef.st_dev(12, monthly_return_data_full)

    s1 = pd.Series(returns_annualized_1, name=("Returns (%)", "1 Year"))
    s2 = pd.Series(returns_annualized_2, name=("Returns (%)", "2 Year"))
    s3 = pd.Series(returns_annualized_5, name=("Returns (%)", "5 Year"))
    s4 = pd.Series(returns_annualized_10, name=("Returns (%)", "10 Year"))
    s5 = pd.Series(vol_annualized_1, name=("Volatility (Standard Deviation(%))", "1 Year"))
    s6 = pd.Series(vol_annualized_2, name=("Volatility (Standard Deviation(%))", "2 Year"))
    s7 = pd.Series(vol_annualized_5, name=("Volatility (Standard Deviation(%))", "5 Year"))
    s8 = pd.Series(vol_annualized_10, name=("Volatility (Standard Deviation(%))", "10 Year"))

    historical_table = ef.make_table(s1, s2, s3, s4, s5, s6, s7, s8)
#create the historical graph for second page

    asset_data2 = ef.read_data("EF_Data_Summary.csv")
    historical_graph = ef.historical_graph(asset_data2)

#header
    st.header("Historical Data")
#graph
    st.plotly_chart(historical_graph, use_container_width=True)

#table
    st.dataframe(historical_table)

















        