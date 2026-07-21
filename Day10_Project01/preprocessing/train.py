from sklearn.cluster import  DBSCAN
from sklearn.preprocessing import StandardScaler

def get_model(df,eps=1,min_samples=4):
    # Handle Missing values
    df = df.dropna()

    # freature Engineering
    df  = df.drop(columns=['CID','OrderDate','OID'])
    # Convert numeric columns
    for col in ['Amt','Profit','Qty']:
        df[col] = df[col].astype(float)

    # Scaling
    scaler = StandardScaler()
    for col in df.select_dtypes(include=float):
        df[col] = scaler.fit_transform(df[[col]])
    
    # Encoding
    for col in df.select_dtypes(include=object):
        df[col] = df[col].map(lambda val:list(df[col].unique()).index(val))

    # Modeling Training for labels
    dbscan = DBSCAN(eps=eps,
                    min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    return labels