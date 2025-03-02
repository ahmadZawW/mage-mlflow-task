@data_loader
def load_data(*args, **kwargs):

    import pandas as pd
    
    # Load the data
    df = pd.read_csv('mage_pipeline/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Initial preprocessing
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    
    # Handle missing values
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
    df.fillna(df["TotalCharges"].mean(), inplace=True)
    
    return df