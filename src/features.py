import pandas as pd
import numpy as np
import os

def load_and_process(path):
    # fix for capital letter file
    if not os.path.exists(path):
        alt_path = path.replace("online_retail.csv", "Online_Retail.csv")
        if os.path.exists(alt_path):
            path = alt_path
    
    df = pd.read_csv(path, encoding='latin1')
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary_Qty', 'AvgPrice']
    rfm['Monetary'] = df.groupby('CustomerID').apply(lambda x: (x['Quantity']*x['UnitPrice']).sum()).values
    rfm['Monetary_Log'] = np.log1p(rfm['Monetary'])
    rfm['AvgBasketValue'] = rfm['Monetary'] / rfm['Frequency']
    
    # Churn = agar 90 din se zyada Recency hai to churn
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    return rfm