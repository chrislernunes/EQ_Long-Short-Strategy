import pandas as pd
import datetime
import numpy as np
import mysql.connector



#DB connection string
config = {
        'user':'root',
        'password':'root',
        'host':'localhost',
        'database':'nsedb'        
        }
#Query
sql_select = "select b.timestamp, b.open,b.high,b.low,b.close,b.volume from symbolmaster a left join cmstaging b on a.id = b.symbol_id where a.symbol = %s and timestamp >= %s and timestamp <= %s order by timestamp asc"



def dataProcessing(dataset):
    dataset.volume = np.where(dataset.volume <= 0, np.nan, dataset.volume)
    dataset.open = np.where(dataset.open <= 0, np.nan, dataset.open)
    dataset.high = np.where(dataset.high <= 0, np.nan, dataset.high)
    dataset.low = np.where(dataset.high <= 0, np.nan, dataset.low)
    dataset.close = np.where(dataset.high <= 0, np.nan, dataset.close)
    dataset.dropna(inplace=True)
    #dataset['returns'] = np.log(dataset['close']/dataset['close'].shift(1))
    return dataset

def getData(symbol,start,end):
    try:
        conn = mysql.connector.connect(**config)
        c = conn.cursor(buffered=True)
        c.execute(sql_select, [symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')])
        data = pd.DataFrame(c.fetchall())
        data.columns = c.column_names
        data.timestamp = pd.to_datetime(data.timestamp)
        data.set_index('timestamp', inplace=True)
        data = dataProcessing(data)
        data.columns = ['Open','High','Low','Close','Volume']
        return data
        c.close()
        conn.close()
    except mysql.connector.Error as err:
        print("DB Error: " + str(err))
        c.close()
        conn.close()




