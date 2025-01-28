# to run async code in jupyter notebook
import nest_asyncio
nest_asyncio.apply()

# Replace these values with your actual API credentials
client_id = '******'
secret_key = '*******'

strategy_name='option_buying_pivot'

#strategy parameters
index_name='NIFTYBANK'
exchange='NSE'
ticker=f"{exchange}:{index_name}-INDEX"
# ticker='MCX:CRUDEOIL24DECFUT'
strike_count=10
strike_diff=100
account_type='PAPER'

if exchange=='NSE':
    time_zone="Asia/Kolkata"


start_hour,start_min=9,30
end_hour,end_min=15,15
quantity=75

buffer=5
profit_loss_point=30




# Import the required module from the fyers_apiv3 package
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import pandas as pd
import pendulum as dt
import asyncio
import pickle
import time
import webbrowser
import os
import sys
import certifi
import pytz


#python 3.12.8
#pip3 install fyers-apiv3
#pip3 install pandas
#pip3 install setuptools
#pip3 install pendulum


# Bank Nifty Strategy

# Strategy Start Time: 9:30 AM
# Setup:
# Calculate the Pivot, Support, and Resistance levels before the market opens.

# Entry:
# Monitor the spot price of Bank Nifty.
# When the spot price touches any of the calculated levels (Pivot, Support, or Resistance),
# place a limit order for the At-The-Money (ATM) Call or Put option with a 30-point buffer.

# Wait for Execution: Only one of the orders (Call or Put) should get executed.
# We will not take positions in both Call and Put simultaneously.

# Exit Rules:
# Set a Stop Loss and Take Profit of 30 points.
# The trade will end either with a 30-point loss or a 30-point profit.

# Trade Limit:
# Only 1 trade per day is allowed under this strategy.

#for windows ssl error
os.environ['SSL_CERT_FILE'] = certifi.where()


#disable fyersApi and Fyers Request logs
import logging

#disable logging for fyersApi
fyersModel.logging.getLogger().setLevel(logging.CRITICAL)

#logging to file
logging.basicConfig(level=logging.INFO, filename=f'{strategy_name}_{dt.now(time_zone).date()}.log',filemode='a',format="%(asctime)s - %(message)s")


# Check if access.txt exists, then read the file and get the access token
if os.path.exists(f'access-{dt.now(time_zone).date()}.txt'):
    print('access token exists')
    with open(f'access-{dt.now(time_zone).date()}.txt', 'r') as f:
        access_token = f.read()

else:
    # Define response type and state for the session
    response_type = "code"
    state = "sample_state"
    try:
        # Create a session model with the provided credentials
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type=response_type
        )

        # Generate the auth code using the session model
        response = session.generate_authcode()

        # Print the auth code received in the response
        print(response)

        # Open the auth code URL in a new browser window
        webbrowser.open(response, new=1)
        newurl = input("Enter the url: ")
        auth_code = newurl[newurl.index('auth_code=')+10:newurl.index('&state')]

        # Define grant type for the session
        grant_type = "authorization_code"
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type=response_type,
            grant_type=grant_type
        )

        # Set the authorization code in the session object
        session.set_token(auth_code)

        # Generate the access token using the authorization code
        response = session.generate_token()

        # Save the access token to access.txt
        access_token = response["access_token"]
        with open(f'access-{dt.now(time_zone).date()}.txt', 'w') as k:
            k.write(access_token)
    except Exception as e:
        # Print the exception and response for debugging
        print(e, response)
        print('unable to get access token')
        sys.exit()

# Print the access token
print('access token:', access_token)


# Get the current time
current_time=dt.now(time_zone)
start_time=dt.datetime(current_time.year,current_time.month,current_time.day,start_hour,start_min,tz=time_zone)
end_time=dt.datetime(current_time.year,current_time.month,current_time.day,end_hour,end_min,tz=time_zone)
print('start time:', start_time)
print('end time:', end_time)


# Initialize FyersModel instances for synchronous and asynchronous operations
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path=None)
fyers_asysc = fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token, log_path=None)


# Define the data for the option chain request
data = {
    "symbol": ticker,
    "strikecount": strike_count,
    "timestamp": ""
}

# Get the expiry data from the option chain
response = fyers.optionchain(data=data)['data']
expiry = response['expiryData'][0]['date']
print("current_expiry selected", expiry)
expiry_e = response['expiryData'][0]['expiry']

# Define the data for the option chain request with expiry
data = {
    "symbol": ticker,
    "strikecount": strike_count,
    "timestamp": expiry_e
}

# Get the option chain data
response = fyers.optionchain(data=data)['data']
option_chain = pd.DataFrame(response['optionsChain'])
symbols = option_chain['symbol'].to_list()

# Get the current spot price
spot_price = option_chain['ltp'].iloc[0]
print('current spot price is', spot_price)

# Separate the symbols into call and put lists
call_list = []
put_list = []
for s in symbols:
    if s.endswith('CE'):
        call_list.append(s)
    else:
        put_list.append(s)

# Combine the put and call lists
symbols = put_list + call_list
print(symbols)

# Initialize the DataFrame for storing option data
df = pd.DataFrame(columns=['name', 'ltp', 'ch', 'chp', 'avg_trade_price', 'open_price', 'high_price', 'low_price', 'prev_close_price', 'vol_traded_today', 'oi', 'pdoi', 'oipercent', 'bid_price', 'ask_price', 'last_traded_time', 'exch_feed_time', 'bid_size', 'ask_size', 'last_traded_qty', 'tot_buy_qty', 'tot_sell_qty', 'lower_ckt', 'upper_ckt', 'type', 'symbol', 'expiry'])
df['name'] = symbols
df.set_index('name', inplace=True)
print(df)



f = dt.now(time_zone).date() - dt.duration( days=5)
p = dt.now(time_zone).date()

data = {
    "symbol": ticker,
    "resolution": "D",
    "date_format": "1",
    "range_from": f.strftime('%Y-%m-%d'),
    "range_to": p.strftime('%Y-%m-%d'),
    "cont_flag": "1"
}


# Fetch historical data
response2 =fyers.history(data=data)
hist_data = pd.DataFrame(response2['candles'])
hist_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

ist = pytz.timezone('Asia/Kolkata')
hist_data['date'] = pd.to_datetime(hist_data['date'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
# print(hist_data)
hist_data=hist_data[hist_data['date'].dt.date<dt.now(time_zone).date()]
print(hist_data)

#calculate pivot

def calculate_cpr(high, low, close):
    """
    Calculate CPR levels.

    Parameters:
    high (float): The high price.
    low (float): The low price.
    close (float): The close price.

    Returns:
    dict: A dictionary containing Pivot, TC, and BC levels.
    """
    pivot = (high + low + close) / 3

    # Resistance Levels
    r1 = (2 * pivot) - low

    # Support Levels
    s1 = (2 * pivot) - high

    return int(pivot),int(r1),int(s1)

pivot,resistance,support = calculate_cpr(hist_data['high'].iloc[-1], hist_data['low'].iloc[-1], hist_data['close'].iloc[-1])
print('pivot:', pivot, 'resistance:', resistance, 'support:', support)


# Function to get the OTM option based on spot price and side (CE/PE)
def get_otm_option(spot_price, side, points=100):
    if side == 'CE':
        otm_strike = (round(spot_price / strike_diff) * strike_diff) + points
    else:
        otm_strike = (round(spot_price / strike_diff) * strike_diff) - points
    otm_option = option_chain[(option_chain['strike_price'] == otm_strike) & (option_chain['option_type'] == side)]['symbol'].squeeze()
    return otm_option, otm_strike



call_option, call_buy_strike = get_otm_option(spot_price, 'CE', 0)
put_option, put_buy_strike = get_otm_option(spot_price, 'PE', 0)
print('call option:', call_option)
print('put option:', put_option)

# Log the start of the strategy
logging.info('started')


# Function to store data using pickle
def store(data, account_type):
    pickle.dump(data, open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'wb'))

# Function to load data using pickle
def load(account_type):
    return pickle.load(open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'rb'))

# Function to place a limit order
def take_limit_position(ticker, action, quantity, limit_price):
    try:
        data = {
            "symbol": ticker,
            "qty": quantity,
            "type": 1,
            "side": action,
            "productType": "INTRADAY",
            "limitPrice": limit_price,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        response3 = fyers.place_order(data=data)
        logging.info(response3)
        print(response3)
    except Exception as e:
        logging.info(e)
        print(e)
        print('unable to place order for some reason')

# Load or initialize paper trading information
if account_type == 'PAPER':
    try:
        paper_info = load(account_type)
    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        paper_info = {  'call_buy':{'option_name':call_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':quantity,'pnl':0},
                        'put_buy':{'option_name':put_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':quantity,'pnl':0},
                        'condition':False
                    }

# Load or initialize live trading information
else:
    try:
        live_info = load(account_type)
    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        live_info = {  'call_buy':{'option_name':call_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':quantity,'pnl':0},
                        'put_buy':{'option_name':put_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':quantity,'pnl':0},
                        'condition':False
                    }


def paper_order():
    global quantity
    global paper_info
    global df
    global spot_price

    # Get the current spot price
    spot_price = df.loc[ticker, 'ltp']

    print(f"Spot price: {spot_price}, Pivot: {pivot}, Support: {support}, Resistance: {resistance}")

    # Get the current time
    ct = dt.now(time_zone)

    # Check if the current time is greater than the start time
    if ct > start_time:

        # Get trade flags
        call_flag = paper_info['call_buy']['trade_flag']
        put_flag = paper_info['put_buy']['trade_flag']

        # Get stop prices
        call_stop_price = paper_info['call_buy']['current_stop_price']
        put_stop_price = paper_info['put_buy']['current_stop_price']

        # Get target prices
        call_profit_price = paper_info['call_buy']['current_profit_price']
        put_profit_price = paper_info['put_buy']['current_profit_price']

        # Get option names
        call_name = paper_info['call_buy']['option_name']
        put_name = paper_info['put_buy']['option_name']

        # Get buy prices
        call_buy_price = paper_info['call_buy']['buy_price']
        put_buy_price = paper_info['put_buy']['buy_price']

        # Get current prices
        call_price = df.loc[call_name, 'ltp']
        put_price = df.loc[put_name, 'ltp']

        # Get condition
        condition = paper_info['condition']
        print(call_price, put_price)

        # Check if the current time is greater than the end time
        if ct > end_time:
            print('Closing all positions due to end time')

            # Close call buy position if trade flag is set
            if call_flag == 1:
                paper_info['call_buy']['quantity'] = 0  # Update quantity
                call_buy_ltp = df.loc[call_name, 'ltp']  # Get current price
                a = [call_name, call_buy_ltp, 'SELL', 0, 0, spot_price, 0]
                paper_info['call_buy']['filled_df'].loc[ct] = a  # Update dataframe
                paper_info['call_buy']['trade_flag'] = 2  # Update flag
                logging.info('Closing call leg due to end time')

            # Close put buy position if trade flag is set
            if put_flag == 1:
                paper_info['put_buy']['quantity'] = 0  # Update quantity
                put_buy_ltp = df.loc[put_name, 'ltp']  # Get current price
                a = [put_name, put_buy_ltp, 'SELL', 0, 0, spot_price, 0]
                paper_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
                paper_info['put_buy']['trade_flag'] = 2  # Update flag
                logging.info('Closing put leg due to end time')

        # Check if buy condition is satisfied
        if (pivot - buffer <= spot_price <= pivot + buffer) or (support - buffer <= spot_price <= support + buffer) or (resistance - buffer <= spot_price <= resistance + buffer):

            if not condition:
                logging.info('Strategy condition satisfied')

                # Get OTM option names
                call_name, _ = get_otm_option(spot_price, 'CE', 0)
                put_name, _ = get_otm_option(spot_price, 'PE', 0)

                # Update option names and quantities
                paper_info['call_buy']['option_name'] = call_name
                paper_info['call_buy']['quantity'] = quantity
                paper_info['put_buy']['option_name'] = put_name
                paper_info['put_buy']['quantity'] = quantity

                # Get current prices and calculate buy, stop, and profit prices
                call_buy_ltp = df.loc[call_name, 'ltp'] + profit_loss_point
                call_stop_price = call_buy_ltp - profit_loss_point
                call_profit_price = call_buy_ltp + profit_loss_point

                put_buy_ltp = df.loc[put_name, 'ltp'] + profit_loss_point
                put_stop_price = put_buy_ltp - profit_loss_point
                put_profit_price = put_buy_ltp + profit_loss_point

                # Update buy, stop, and profit prices in paper_info
                paper_info['call_buy']['buy_price'] = call_buy_ltp
                paper_info['call_buy']['current_stop_price'] = call_stop_price
                paper_info['call_buy']['current_profit_price'] = call_profit_price

                paper_info['put_buy']['buy_price'] = put_buy_ltp
                paper_info['put_buy']['current_stop_price'] = put_stop_price
                paper_info['put_buy']['current_profit_price'] = put_profit_price

                paper_info['condition'] = True  # Update condition
                logging.info(f"Call price: {call_buy_ltp}, Put price: {put_buy_ltp}")
                print('Done fetching prices')

        # Check call buy condition
        if condition and call_buy_price <= call_price and call_flag == 0:
            a = [call_name, call_price, 'BUY', call_stop_price, call_profit_price, spot_price, quantity]
            paper_info['call_buy']['filled_df'].loc[ct] = a  # Save to dataframe
            paper_info['call_buy']['trade_flag'] = 1  # Update call flag
            paper_info['put_buy']['trade_flag'] = 3  # Update put flag
            logging.info(f'Call buy condition satisfied: {call_name} at {call_price}')

        # Check call sell condition
        elif condition and call_flag == 1:
            if call_price > call_profit_price or call_price < call_stop_price:
                paper_info['call_buy']['quantity'] = 0  # Update quantity
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
                paper_info['call_buy']['filled_df'].loc[ct] = a  # Update dataframe
                paper_info['call_buy']['trade_flag'] = 2  # Update flag
                logging.info(f'Call sell condition satisfied: {call_name} at {call_price}')

        # Check put buy condition
        if condition and put_buy_price <= put_price and put_flag == 0:
            a = [put_name, put_price, 'BUY', put_stop_price, put_profit_price, spot_price, quantity]
            paper_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
            paper_info['put_buy']['trade_flag'] = 1  # Update put flag
            paper_info['call_buy']['trade_flag'] = 3  # Update call flag
            logging.info(f'Put buy condition satisfied: {put_name} at {put_price}')
            print(f'Put buy condition satisfied: {put_name} at {put_price}')

        # Check put sell condition
        elif condition and put_flag == 1:
            if put_price > put_profit_price or put_price < put_stop_price:
                paper_info['put_buy']['quantity'] = 0  # Update quantity
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
                paper_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
                paper_info['put_buy']['trade_flag'] = 2  # Update flag
                print('Put sell condition satisfied')

        # Save filled dataframes to CSV files
        if not paper_info['call_buy']['filled_df'].empty:
            paper_info['call_buy']['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        if not paper_info['put_buy']['filled_df'].empty:
            paper_info['put_buy']['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        # Store paper_info using pickle
        store(paper_info, account_type)


def real_order():
    global quantity
    global live_info
    global df
    global spot_price

    # Get the current spot price
    spot_price = df.loc[ticker, 'ltp']

    print(f"Spot price: {spot_price}, Pivot: {pivot}, Support: {support}, Resistance: {resistance}")

    # Get the current time
    ct = dt.now(time_zone)

    # Check if the current time is greater than the start time
    if ct > start_time:

        # Get trade flags
        call_flag = live_info['call_buy']['trade_flag']
        put_flag = live_info['put_buy']['trade_flag']

        # Get stop prices
        call_stop_price = live_info['call_buy']['current_stop_price']
        put_stop_price = live_info['put_buy']['current_stop_price']

        # Get target prices
        call_profit_price = live_info['call_buy']['current_profit_price']
        put_profit_price = live_info['put_buy']['current_profit_price']

        # Get option names
        call_name = live_info['call_buy']['option_name']
        put_name = live_info['put_buy']['option_name']

        # Get buy prices
        call_buy_price = live_info['call_buy']['buy_price']
        put_buy_price = live_info['put_buy']['buy_price']

        # Get current prices
        call_price = df.loc[call_name, 'ltp']
        put_price = df.loc[put_name, 'ltp']

        # Get condition
        condition = live_info['condition']
        print(call_price, put_price)

        # Check if the current time is greater than the end time
        if ct > end_time:
            print('Closing all positions due to end time')

            # Close call buy position if trade flag is set
            if call_flag == 1:
                live_info['call_buy']['quantity'] = 0  # Update quantity
                call_buy_ltp = df.loc[call_name, 'ltp']  # Get current price
                a = [call_name, call_buy_ltp, 'SELL', 0, 0, spot_price, 0]
                live_info['call_buy']['filled_df'].loc[ct] = a  # Update dataframe
                live_info['call_buy']['trade_flag'] = 2  # Update flag
                logging.info('Closing call leg due to end time')
                data = {"id": call_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)
                logging.info(f'Closed put sell position: {call_name} at {call_price}')

            # Close put buy position if trade flag is set
            if put_flag == 1:
                live_info['put_buy']['quantity'] = 0  # Update quantity
                put_buy_ltp = df.loc[put_name, 'ltp']  # Get current price
                a = [put_name, put_buy_ltp, 'SELL', 0, 0, spot_price, 0]
                live_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
                live_info['put_buy']['trade_flag'] = 2  # Update flag
                logging.info('Closing put leg due to end time')
                data = {"id": put_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)
                logging.info(f'Closed put sell position: {put_name} at {put_price}')

        # Check if buy condition is satisfied
        if (pivot - buffer <= spot_price <= pivot + buffer) or (support - buffer <= spot_price <= support + buffer) or (resistance - buffer <= spot_price <= resistance + buffer):

            if not condition:
                logging.info('Strategy condition satisfied')

                # Get OTM option names
                call_name, _ = get_otm_option(spot_price, 'CE', 0)
                put_name, _ = get_otm_option(spot_price, 'PE', 0)

                # Update option names and quantities
                live_info['call_buy']['option_name'] = call_name
                live_info['call_buy']['quantity'] = quantity
                live_info['put_buy']['option_name'] = put_name
                live_info['put_buy']['quantity'] = quantity

                # Get current prices and calculate buy, stop, and profit prices
                call_buy_ltp = df.loc[call_name, 'ltp'] + profit_loss_point
                call_stop_price = call_buy_ltp - profit_loss_point
                call_profit_price = call_buy_ltp + profit_loss_point

                put_buy_ltp = df.loc[put_name, 'ltp'] + profit_loss_point
                put_stop_price = put_buy_ltp - profit_loss_point
                put_profit_price = put_buy_ltp + profit_loss_point

                # Update buy, stop, and profit prices in live_info
                live_info['call_buy']['buy_price'] = call_buy_ltp
                live_info['call_buy']['current_stop_price'] = call_stop_price
                live_info['call_buy']['current_profit_price'] = call_profit_price

                live_info['put_buy']['buy_price'] = put_buy_ltp
                live_info['put_buy']['current_stop_price'] = put_stop_price
                live_info['put_buy']['current_profit_price'] = put_profit_price

                live_info['condition'] = True  # Update condition
                logging.info(f"Call price: {call_buy_ltp}, Put price: {put_buy_ltp}")
                print('Done fetching prices')

        # Check call buy condition
        if condition and call_buy_price <= call_price and call_flag == 0:
            a = [call_name, call_price, 'BUY', call_stop_price, call_profit_price, spot_price, quantity]
            take_limit_position(call_name, 1, quantity, call_buy_price)
            live_info['call_buy']['filled_df'].loc[ct] = a  # Save to dataframe
            live_info['call_buy']['trade_flag'] = 1  # Update call flag
            live_info['put_buy']['trade_flag'] = 3  # Update put flag
            logging.info(f'Call buy condition satisfied: {call_name} at {call_price}')

        # Check call sell condition
        elif condition and call_flag == 1:
            if call_price > call_profit_price or call_price < call_stop_price:
                live_info['call_buy']['quantity'] = 0  # Update quantity
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
                live_info['call_buy']['filled_df'].loc[ct] = a  # Update dataframe
                live_info['call_buy']['trade_flag'] = 2  # Update flag
                logging.info(f'Call sell condition satisfied: {call_name} at {call_price}')
                data = {"id": call_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)
                logging.info(f'Closed put sell position: {call_name} at {call_price}')


        # Check put buy condition
        if condition and put_buy_price <= put_price and put_flag == 0:
            a = [put_name, put_price, 'BUY', put_stop_price, put_profit_price, spot_price, quantity]
            take_limit_position(put_name, 1, quantity, put_price)
            live_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
            live_info['put_buy']['trade_flag'] = 1  # Update put flag
            live_info['call_buy']['trade_flag'] = 3  # Update call flag
            logging.info(f'Put buy condition satisfied: {put_name} at {put_price}')
            print(f'Put buy condition satisfied: {put_name} at {put_price}')

        # Check put sell condition
        elif condition and put_flag == 1:
            if put_price > put_profit_price or put_price < put_stop_price:
                live_info['put_buy']['quantity'] = 0  # Update quantity
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
                live_info['put_buy']['filled_df'].loc[ct] = a  # Update dataframe
                live_info['put_buy']['trade_flag'] = 2  # Update flag
                print('Put sell condition satisfied')
                data = {"id": put_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)
                logging.info(f'Closed put sell position: {put_name} at {put_price}')

        # Save filled dataframes to CSV files
        if not live_info['call_buy']['filled_df'].empty:
            live_info['call_buy']['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        if not live_info['put_buy']['filled_df'].empty:
            live_info['put_buy']['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        # Store live_info using pickle
        store(live_info, account_type)






def onmessage(ticks):
    global df
    # print(ticks)
    if ticks.get('symbol'):
        for key,value in ticks.items():
            #updating dataframe
            df.loc[ticks.get('symbol'), key] = value
            df.drop_duplicates(inplace=True)

def onerror(message):
    print("Error:", message)

def onclose(message):
    print("Connection closed:", message)

def onopen():
    global symbols
    # Specify the data type and symbols you want to subscribe to
    data_type = "SymbolUpdate"

    fyers_socket.subscribe(symbols=symbols, data_type=data_type)

    # Keep the socket running to receive real-time data
    fyers_socket.keep_running()
    print('starting socket')


# Create a FyersDataSocket instance with the provided parameters
fyers_socket = data_ws.FyersDataSocket(
    access_token=f"{client_id}:{access_token}",  # Access token in the format "appid:accesstoken"
    log_path=None,  # Path to save logs. Leave empty to auto-create logs in the current directory.
    litemode=False,  # Lite mode disabled. Set to True if you want a lite response.
    write_to_file=False,  # Save response in a log file instead of printing it.
    reconnect=True,  # Enable auto-reconnection to WebSocket on disconnection.
    on_connect=onopen,  # Callback function to subscribe to data upon connection.
    on_close=onclose,  # Callback function to handle WebSocket connection close events.
    on_error=onerror,  # Callback function to handle WebSocket errors.
    on_message=onmessage  # Callback function to handle incoming messages from the WebSocket.
)

fyers_socket.connect()


def chase_order(ord_df):
    # Check if the order dataframe is not empty
    if not ord_df.empty:
        # Filter orders with status 6 (open orders)
        ord_df = ord_df[ord_df['status'] == 6]
        # Iterate through each order in the dataframe
        for i, o1 in ord_df.iterrows():
            # Get the symbol name from the order
            name = o1['symbol']
            # Get the current price of the symbol from the dataframe
            current_price = df.loc[name, 'ltp']
            try:
                # Check if the order type is limit order (type 1)
                if o1['type'] == 1:
                    # Get the order details
                    name = o1['symbol']
                    id1 = o1['id']
                    lmt_price = o1['limitPrice']
                    qty = o1['qty']
                    # Determine the new limit price based on the current price
                    if current_price > lmt_price:
                        new_lmt_price = round(lmt_price + 0.1, 2)
                    else:
                        new_lmt_price = round(lmt_price - 0.1, 2)
                    # Print the order details and new limit price
                    print(name, lmt_price, qty, new_lmt_price)
                    # Modify the order with the new limit price
                    data = {
                        "id": id1,
                        "type": 1,
                        "limitPrice": new_lmt_price,
                        "qty": qty
                    }
                    # Send the modify order request to Fyers
                    response = fyers.modify_order(data=data)
                    # Print the response from Fyers
                    print(response)
            except:
                # Print an error message if there is an exception
                print('error in chasing order')


pnl=0

async def main_strategy_code():
    global df

    while True:
        ct = dt.now(time_zone)  # Get the current time

        #close program 2 min after end time
        if ct > end_time + dt.duration( minutes=2):
            logging.info('closing program')
            sys.exit()

        # Get current PnL and chase order every 5 seconds
        if ct.second in range(0, 59, 5):
            try:
                # Fetch order book information asynchronously
                order_response = await fyers_asysc.orderbook()

                # Convert order book response to DataFrame if it exists
                if order_response['orderBook']:
                    order_df = pd.DataFrame(order_response['orderBook'])
                else:
                    order_df = pd.DataFrame()

                # Chase the order based on the order DataFrame
                chase_order(order_df)

                # Fetch positions asynchronously
                pos1 = await fyers_asysc.positions()

                # Get the total PnL from the positions
                pnl = int(pos1.get('overall').get('pl_total'))

            except:
                # Print error message if unable to fetch PnL or chase order
                print('unable to fetch pnl or chase order')

            # Print the current PnL
            # print("current_pnl", pnl)

        # Run strategy if DataFrame is not empty
        if df.shape[0] != 0:
            print(ct)  # Print the current time

            # Execute paper order or real order based on account type
            if account_type == 'PAPER':
                paper_order()
            else:
                real_order()

        # Sleep for 1 second before the next iteration
        await asyncio.sleep(1)

time.sleep(5)

async def main():
    while True:
        await main_strategy_code()

asyncio.run(main())


#if csv file exist read it
if os.path.exists(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv'):
  df=pd.read_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')
else:
  df=pd.DataFrame()
df