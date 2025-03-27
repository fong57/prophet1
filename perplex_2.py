import yfinance as yf
from prophet import Prophet
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_stock_data():
    # Prompt user for stock ticker
    stock = input("Please enter the stock ticker (e.g., 3988.HK): ")
    
    # Prompt user for start and end dates
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    
    try:
        # Fetch historical data
        data = yf.download(stock, start=start_date)
        
        # Check if data is available
        if data.empty:
            print("No data available for the specified period.")
            return None
        
        # Prepare DataFrame for Prophet
        data = data.reset_index()[['Date', 'Close']].dropna()
        data.columns = ['ds', 'y']  # Direct column renaming
        
        return data, stock
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def forecast_stock(data):
    # Prophet Forecasting
    m = Prophet()
    m.fit(data)
    
    # Create future dataframe
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    
    # Display forecast results
    print("\nForecast Results:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    return forecast

def save_plot(data, forecast, stock):
    """Generate and save forecast visualization"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(data['ds'], data['y'], 'b-', label='Historical Data')
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], 'r--', label='Forecast')
    
    # Add confidence interval
    plt.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'],
                    color='pink', alpha=0.3, label='Confidence Interval')
    
    plt.title(f'{stock} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, f"{stock}_forecast_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nForecast plot saved to: {plot_path}")

def save_to_csv(data, forecast, stock):
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save prepared dataset
    dataset_filename = f"{stock}_dataset.csv"
    dataset_path = os.path.join(current_dir, dataset_filename)
    data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to: {dataset_path}")
    
    # Save forecast results
    forecast_filename = f"{stock}_forecast.csv"
    forecast_path = os.path.join(current_dir, forecast_filename)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_path, index=False)
    print(f"Forecast saved to: {forecast_path}")

def main():
    data, stock = get_stock_data()
    
    if data is not None:
        print("\nPrepared Dataset:")
        print(data.head())
        
        forecast = forecast_stock(data)
        save_to_csv(data, forecast, stock)
        save_plot(data, forecast, stock)  # Add this line

if __name__ == "__main__":
    main()
