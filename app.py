from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


pd.set_option('future.no_silent_downcasting', True)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure secret key for session management

# Allowed tickers can be populated from a reliable source or database

ALLOWED_TICKERS = set(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'NVDA', 'TSLA'])  # Example tickers

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper()
        if ticker in ALLOWED_TICKERS:
            return redirect(url_for('analyze', ticker=ticker))
        else:
            flash('Invalid or unsupported ticker symbol. Please try again.')
            return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/analyze/<ticker>')
def analyze(ticker):
    try:
        # Fetch financial data
        data = fetch_financial_data(ticker)

        # Calculate financial ratios
        ratios = calculate_financial_ratios(data)

        # Generate financial projections
        projections = generate_projections(data)

        # Prepare earnings data for the chart
        earnings_df = data['earnings']
        earnings_years = earnings_df['Year'].astype(str).tolist()
        earnings_values = earnings_df['Earnings'].astype(float).tolist()

        # Append projection
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
            header=False  # Do not include the header since index contains the ratio names
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
        flash(f"Data unavailable for {ticker}: {str(e)}")
        return redirect(url_for('home'))

    except KeyError as e:
        flash(f"Data parsing error for {ticker}: {str(e)}")
        return redirect(url_for('home'))

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}")
        return redirect(url_for('home'))


def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.get_financials()
    balance_sheet = stock.get_balance_sheet()
    cashflow = stock.get_cashflow()
    income_stmt = stock.get_income_stmt()

    # Debug statements
    print(f"Ticker: {ticker}")
    print("Income Statement Index:")
    print(income_stmt.index.tolist())
    print("Income Statement Columns:")
    print(income_stmt.columns.tolist())

    # Check if income_stmt is not None and not empty
    if income_stmt is not None and not income_stmt.empty:
        # Attempt to extract net income
        possible_net_income_labels = [
            'NetIncome',
            'NetIncomeCommonStockholders',
            'NetIncomeIncludingNoncontrollingInterests',
            'NetIncomeContinuousOperations',
            'NetIncomeFromContinuingOperationNetMinorityInterest',
            'NetIncomeFromContinuingAndDiscontinuedOperation',
            'NormalizedIncome',
            'DilutedNIAvailtoComStockholders',
        ]
        net_income = None
        for label in possible_net_income_labels:
            if label in income_stmt.index:
                net_income = income_stmt.loc[label]
                print(f"Found Net Income under label: {label}")
                break
        if net_income is None:
            raise KeyError("Net Income label not found in income statement.")
        # Convert net income Series to DataFrame for consistency
        earnings = pd.DataFrame({'Earnings': net_income})
        earnings.index.name = 'Year'
        earnings.reset_index(inplace=True)
    else:
        earnings = None

    # Check if essential data is available
    if (financials is None or financials.empty or
        balance_sheet is None or balance_sheet.empty or
        earnings is None or earnings.empty):
        raise ValueError("Essential financial data is unavailable for this ticker.")

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

    # Debug statements
    print("Financials Index:")
    print(financials.index.tolist())
    print("Balance Sheet Index:")
    print(balance_sheet.index.tolist())

    # Ensure data is filled
    financials = financials.fillna(0).infer_objects()
    balance_sheet = balance_sheet.fillna(0).infer_objects()

    # Extract required data
    try:
        # Extract data using adjusted labels
        current_assets = balance_sheet.loc['CurrentAssets']
        current_liabilities = balance_sheet.loc['CurrentLiabilities']
        total_assets = balance_sheet.loc['TotalAssets']

        # Total Liabilities
        if 'TotalLiabilitiesNetMinorityInterest' in balance_sheet.index:
            total_liabilities = balance_sheet.loc['TotalLiabilitiesNetMinorityInterest']
        else:
            total_liabilities = balance_sheet.loc['CurrentLiabilities'] + balance_sheet.loc['TotalNonCurrentLiabilitiesNetMinorityInterest']

        # Total Equity
        if 'StockholdersEquity' in balance_sheet.index:
            total_equity = balance_sheet.loc['StockholdersEquity']
        elif 'CommonStockEquity' in balance_sheet.index:
            total_equity = balance_sheet.loc['CommonStockEquity']
        else:
            raise KeyError("Total equity label not found in balance sheet.")

        net_income = financials.loc['NetIncome']
        revenue = financials.loc['TotalRevenue']

        # Calculate ratios and take the mean (or appropriate aggregation)
        ratios_dict = {
            'Current Ratio': (current_assets / current_liabilities).mean(),
            'Debt to Equity Ratio': (total_liabilities / total_equity).mean(),
            'Return on Assets': (net_income / total_assets).mean(),
            'Net Profit Margin': (net_income / revenue).mean(),
        }

        # Convert the dictionary to a DataFrame
        ratios = pd.DataFrame.from_dict(ratios_dict, orient='index', columns=['Value'])

        # Return the ratios DataFrame
        return ratios
    except KeyError as e:
        print(f"KeyError in calculate_financial_ratios: {e}")
        raise KeyError(f"Missing data for ratio calculation: {e}")




def generate_projections(data):
    # Use the 'Earnings' DataFrame to create projections
    earnings_df = data['earnings']
    earnings_df['Year'] = pd.to_datetime(earnings_df['Year'], format='%Y')
    earnings_df.sort_values('Year', inplace=True)
    
    # Prepare data for linear regression
    earnings_df['YearOrdinal'] = earnings_df['Year'].map(pd.Timestamp.toordinal)
    X = earnings_df[['YearOrdinal']]
    y = earnings_df['Earnings']
    
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
