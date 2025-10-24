import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/raw/crop_data.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
   
    df = df.dropna() 
    
 
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])  
    
    return df, le

def split_data(df, test_size=0.2, random_state=42):
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
   
    X_train.join(y_train).to_csv("data/processed/train.csv", index=False)
    X_test.join(y_test).to_csv("data/processed/test.csv", index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    df, le = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Data preprocessing completed. Train/Test CSVs saved in data/processed/")
