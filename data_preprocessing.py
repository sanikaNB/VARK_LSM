# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Year'] = le.fit_transform(df['Year'])
    df['Brain Dominance'] = le.fit_transform(df['Brain Dominance'])
    df['Primary VARK'] = le.fit_transform(df['Primary VARK'])

    X = df.iloc[:, 6:-8] 
    y = df['Primary VARK']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test





# # data_preprocessing.py
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# def preprocess_data(file_path):
#     df = pd.read_csv(file_path)

#     le = LabelEncoder()
#     df['Gender'] = le.fit_transform(df['Gender'])
#     df['Year'] = le.fit_transform(df['Year'])
#     df['Brain Dominance'] = le.fit_transform(df['Brain Dominance'])
#     df['Primary VARK'] = le.fit_transform(df['Primary VARK'])

#     X = df.iloc[:, 6:-1]  # All columns except 'Primary VARK'
#     y = df['Primary VARK']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test
