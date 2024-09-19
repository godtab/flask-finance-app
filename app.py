from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from flask_caching import Cache

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure secret key for session management

# Configure caching to use filesystem-based cache for persistence
cache = Cache(app, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/my_cache_directory',
    'CACHE_DEFAULT_TIMEOUT': 3600  # Cache timeout set to 1 hour
})

# Suppress pandas SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

# Set to True to allow any publicly traded company, False to restrict to S&P 500
ALLOW_ALL_TICKERS = True

# Fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"Fetched S&P 500 Tickers: {tickers}")  # Debugging
    return set(tickers)

# Initialize ALLOWED_TICKERS based on your preference
if ALLOW_ALL_TICKERS:
    ALLOWED_TICKERS = None  # Allow all tickers
else:
    ALLOWED_TICKERS = get_sp500_tickers()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        print(f"Received ticker: {ticker}")  # Debugging: Log the ticker submitted

        if ALLOWED_TICKERS is not None:
            print(f"ALLOWED_TICKERS contains {len(ALLOWED_TICKERS)} tickers.")  # Debugging
            if ticker in ALLOWED_TICKERS:
                print(f"{ticker} is in ALLOWED_TICKERS")  # Debugging: Ticker passed validation
                return redirect(url_for('analyze', ticker=ticker))
            else:
                print(f"{ticker} is not in ALLOWED_TICKERS")  # Debugging: Ticker failed validation
                flash('Invalid or unsupported ticker symbol. Please try again.')
                return redirect(url_for('home'))
        else:
            if is_valid_ticker(ticker):
                print(f"{ticker} is a valid ticker")  # Debugging: Ticker passed validation
                return redirect(url_for('analyze', ticker=ticker))
            else:
                print(f"{ticker} is not valid")  # Debugging: Ticker failed validation
                flash('Invalid or unsupported ticker symbol. Please try again.')
                return redirect(url_for('home'))
    return render_template('index.html')

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period='1d')
        if history.empty:
            print(f"No historical data found for ticker {ticker}")
            return False
        info = stock.info
        print(f"Ticker Info for {ticker}: {info}")  # Debugging: Log the info dictionary

        # Check multiple fields to validate the ticker
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return True
        if 'shortName' in info and info['shortName']:
            return True
        if 'quoteType' in info and info['quoteType'] == 'EQUITY':
            return True
        # Add more checks as necessary
        return False
    except Exception as e:
        print(f"Validation error for ticker {ticker}: {e}")
        return False

@app.route('/analyze/<ticker>')
def analyze(ticker):
    try:
        print(f"Analyzing ticker: {ticker}")  # Debugging: log which ticker is being analyzed

        # Fetch financial data
        data = fetch_financial_data(ticker)

        # Calculate financial ratios
        ratios = calculate_financial_ratios(data)

        # Generate financial projections
        projections = generate_projections(data)

        # Prepare earnings data for the chart
        earnings_df = data['earnings']
        earnings_df['Year'] = earnings_df['Year'].astype(str)
        earnings_df['Earnings'] = pd.to_numeric(earnings_df['Earnings'], errors='coerce')
        earnings_years = earnings_df['Year'].tolist()
        earnings_values = earnings_df['Earnings'].tolist()

        # Append projection if available
        if projections is not None:
            earnings_years.append(str(projections['Next Year']))
            earnings_values.append(float(projections['Projected Earnings']))

        # Prepare chart data
        earnings_chart_data = {
            'labels': earnings_years,
            'datasets': [{
                'label': 'Earnings',
                'data': earnings_values,
                'backgroundColor': 'rgba(54, 162, 235, 0.5)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'fill': False,
                'tension': 0.1
            }]
        }

        # Convert ratios DataFrame to HTML
        ratios_html = ratios.to_html(
            classes='table table-striped',
            border=0,
            header=True  # Include headers for clarity
        )

        # Render the analysis page with fetched data
        return render_template(
            'analysis.html',
            ticker=ticker,
            ratios=ratios_html,
            projections=projections,
            earnings_chart_data=earnings_chart_data
        )

    except ValueError as e:
        flash(f"Data unavailable for {ticker}: {e}")
        return redirect(url_for('home'))

    except KeyError as e:
        flash(f"Data parsing error for {ticker}: {e}")
        return redirect(url_for('home'))

    except Exception as e:
        flash(f"An unexpected error occurred for {ticker}: {e}")
        return redirect(url_for('home'))

@cache.memoize(timeout=3600)  # Cache results for 1 hour
def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        # income_stmt is same as financials
        income_stmt = financials
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")

    # Check if essential data is available
    if any(df is None or df.empty for df in [financials, balance_sheet, cashflow, income_stmt]):
        raise ValueError("Essential financial data is unavailable for this ticker.")

    # Define possible labels for Net Income
    possible_net_income_labels = [
        'Net Income',
        'net_income',
        'Net income',
        'net Income',
        'netincome',
        'NetIncome',
        'NetIncomeCommonStockholders',
        'NetIncomeIncludingNoncontrollingInterests',
        'NetIncomeContinuousOperations',
        'NetIncomeFromContinuingOperationNetMinorityInterest',
        'NetIncomeFromContinuingAndDiscontinuedOperation',
        'NormalizedIncome',
        'DilutedNIAvailtoComStockholders',
    ]

    # Attempt to extract Net Income using the helper function
    net_income_series = get_first_available_label(income_stmt, possible_net_income_labels)
    if net_income_series is None:
        raise KeyError("Net Income label not found in income statement.")

    print(f"Found Net Income under label: {net_income_series.name}")  # Debugging

    # Convert net income Series to DataFrame for consistency
    earnings = pd.DataFrame({'Earnings': net_income_series})
    earnings.index.name = 'Year'
    earnings.reset_index(inplace=True)

    data = {
        'financials': financials,
        'balance_sheet': balance_sheet,
        'cashflow': cashflow,
        'earnings': earnings
    }
    return data

def calculate_financial_ratios(data):
    financials = data['financials']
    balance_sheet = data['balance_sheet']

    # Replace zeros with NaN to avoid division by zero
    financials.replace(0, np.nan, inplace=True)
    balance_sheet.replace(0, np.nan, inplace=True)

    # Define possible labels for Total Equity
    possible_total_equity_labels = [
        'TotalStockholderEquity', 
        'totalequity',
        'Total Equity',
        'Totalequity',
        'Total_Equity',
        'Total_equity',
        'total equity',
        'StockholdersEquity',
        'Total Stockholder Equity', 
        'Total Stockholders Equity',
        'CommonStockEquity',
        'Equity',
        'TotalStockholdersEquity',
        'Total Shareholders Equity',
        'TotalEquity',
        'Equity attributable to shareholders'
    ]

    # Define possible labels for Current Assets
    possible_current_assets_labels = [
        'Current Assets',
        'Total Current Assets',
        'TotalAssetsCurrent',
        'AssetsCurrent',
        'TotalCurrentAssets'
    ]

    # Define possible labels for Current Liabilities
    possible_current_liabilities_labels = [
        'Current Liabilities',
        'Total Current Liabilities',
        'CurrentLiabilities',
        'LiabilitiesCurrent',
        'TotalCurrentLiabilities'
    ]

    # Define possible labels for Total Assets
    possible_total_assets_labels = [
        'Total Assets',
        'Assets',
        'TotalAssets',
        'Total assets',
        'AssetTotal'
    ]

    # Define possible labels for Total Liabilities
    possible_total_liab_labels = [
        'Total Liab',
        'Total Liabilities',
        'TotalLiab',
        'Liabilities',
        'Total liabilities'
    ]

    # Define possible labels for Long Term Debt
    possible_long_term_debt_labels = [
        'Long Term Debt',
        'LongTermDebt',
        'DebtLongTerm',
        'LongTermBorrowings',
        'DebtNonCurrent'
    ]

    # Extract required data using the helper function
    current_assets = get_first_available_label(balance_sheet, possible_current_assets_labels)
    if current_assets is None:
        raise KeyError("Current Assets label not found in balance sheet.")

    current_liabilities = get_first_available_label(balance_sheet, possible_current_liabilities_labels)
    if current_liabilities is None:
        raise KeyError("Current Liabilities label not found in balance sheet.")

    total_assets = get_first_available_label(balance_sheet, possible_total_assets_labels)
    if total_assets is None:
        raise KeyError("Total Assets label not found in balance sheet.")

    # Extract Total Liabilities
    total_liabilities = get_first_available_label(balance_sheet, possible_total_liab_labels)
    if total_liabilities is None:
        # Attempt to sum Current Liabilities and Long Term Debt
        long_term_debt = get_first_available_label(balance_sheet, possible_long_term_debt_labels)
        if long_term_debt is not None:
            total_liabilities = current_liabilities + long_term_debt
            print("Calculated Total Liabilities by summing Current Liabilities and Long Term Debt.")  # Debugging
        else:
            raise KeyError("Total Liabilities label not found in balance sheet, and Long Term Debt is unavailable.")

    # Extract Total Equity
    possible_total_equity_labels = [
        'TotalStockholderEquity', 
        'totalequity',
        'Total Equity',
        'Totalequity',
        'Total_Equity',
        'Total_equity',
        'total equity',
        'StockholdersEquity',
        'Total Stockholder Equity', 
        'Total Stockholders Equity',
        'CommonStockEquity',
        'Equity',
        'TotalStockholdersEquity',
        'Total Shareholders Equity',
        'TotalEquity',
        'Equity attributable to shareholders'
    ]
    total_equity = get_first_available_label(balance_sheet, possible_total_equity_labels)
    if total_equity is None:
        raise KeyError("Total equity label not found in balance sheet.")

    # Extract Net Income and Revenue using the helper function
    possible_net_income_labels = [
        'Net Income',
        'net_income',
        'Net income',
        'net Income',
        'netincome',
        'NetIncome',
        'NetIncomeCommonStockholders',
        'NetIncomeIncludingNoncontrollingInterests',
        'NetIncomeContinuousOperations',
        'NetIncomeFromContinuingOperationNetMinorityInterest',
        'NetIncomeFromContinuingAndDiscontinuedOperation',
        'NormalizedIncome',
        'DilutedNIAvailtoComStockholders',
    ]
    net_income = get_first_available_label(financials, possible_net_income_labels)
    if net_income is None:
        raise KeyError("Net Income label not found in financials.")

    possible_revenue_labels = [
        'Total Revenue',
        'Revenue',
        'TotalRevenue',
        'Sales',
        'Total Sales',
        'Operating Revenue',
        'Revenue from Operations'
    ]
    revenue = get_first_available_label(financials, possible_revenue_labels)
    if revenue is None:
        raise KeyError("Revenue label not found in financials.")

    # Avoid division by zero and handle NaN values
    current_liabilities.replace(0, np.nan, inplace=True)
    total_equity.replace(0, np.nan, inplace=True)
    total_assets.replace(0, np.nan, inplace=True)
    revenue.replace(0, np.nan, inplace=True)

    # Calculate ratios
    ratios_dict = {
        'Current Ratio': (current_assets / current_liabilities).mean(skipna=True),
        'Debt to Equity Ratio': (total_liabilities / total_equity).mean(skipna=True),
        'Return on Assets': (net_income / total_assets).mean(skipna=True),
        'Net Profit Margin': (net_income / revenue).mean(skipna=True),
    }

    # Replace NaN or infinite values with 'N/A'
    for key in ratios_dict:
        value = ratios_dict[key]
        if pd.isnull(value) or np.isinf(value):
            ratios_dict[key] = 'N/A'
        else:
            ratios_dict[key] = round(value, 2)

    # Convert the dictionary to a DataFrame
    ratios = pd.DataFrame.from_dict(ratios_dict, orient='index', columns=['Value'])

    # Return the ratios DataFrame
    return ratios

def generate_projections(data):
    # Use the 'Earnings' DataFrame to create projections
    earnings_df = data['earnings']
    earnings_df['Year'] = pd.to_datetime(earnings_df['Year'], format='%Y', errors='coerce')
    earnings_df['Earnings'] = pd.to_numeric(earnings_df['Earnings'], errors='coerce')
    earnings_df.sort_values('Year', inplace=True)

    # Drop rows with NaN in 'Earnings' or 'Year'
    earnings_df.dropna(subset=['Earnings', 'Year'], inplace=True)

    # Check if there's enough data to fit the model
    if len(earnings_df) < 2:
        # Not enough data to make a projection
        print("Not enough data to fit the model.")
        return None  # Return None or handle appropriately

    # Prepare data for linear regression
    earnings_df['YearOrdinal'] = earnings_df['Year'].map(pd.Timestamp.toordinal)
    X = earnings_df[['YearOrdinal']].values
    y = earnings_df['Earnings'].values

    # Check for NaN values in X and y
    if np.isnan(X).any() or np.isnan(y).any():
        print("NaN values detected in X or y.")
        return None

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Project earnings for the next year
    next_year = earnings_df['Year'].dt.year.max() + 1
    next_year_date = pd.Timestamp(year=next_year, month=1, day=1)
    next_year_ordinal = next_year_date.toordinal()
    projected_earnings = model.predict([[next_year_ordinal]])[0]

    projections = {
        'Next Year': next_year,
        'Projected Earnings': projected_earnings
    }
    return projections

def get_first_available_label(df, possible_labels):
    """
    Helper function to extract the first available label from a DataFrame's index.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to search.
        possible_labels (list): A list of possible labels to search for.
        
    Returns:
        pd.Series or None: The first matching Series if found, else None.
    """
    # Normalize the DataFrame's index labels for flexible matching
    normalized_df_labels = [label.lower().replace(" ", "").replace("_", "") for label in df.index]
    label_mapping = {label.lower().replace(" ", "").replace("_", ""): label for label in df.index}

    for label in possible_labels:
        normalized_label = label.lower().replace(" ", "").replace("_", "")
        if normalized_label in label_mapping:
            actual_label = label_mapping[normalized_label]
            print(f"Matched label '{actual_label}' for '{label}'.")  # Debugging
            return df.loc[actual_label]

    # If no label is found, print available labels for debugging
    print("Available labels in DataFrame:", df.index.tolist())
    return None

# Error handlers
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    flash('An unexpected error occurred. Please try again later.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Run the app in development mode
    app.run(debug=True)
