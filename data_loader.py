import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_file_path):
    df = pd.read_csv('dataset .csv')

    df = df.drop_duplicates()
    df = df.dropna()

    df = df[[
        'Hour',
        'Patient_Inflow',
        'Emergency_Cases',
        'Bed_Occupancy_Rate',
        'Surgery_Count',
        'Operating_Theater_Utilization',
        'Holiday_Flag',
        'Doctors_Scheduled',
        'Adjusted_Doctors'
    ]]

    df = df.astype(int)

    state_columns = [
        'Hour',
        'Patient_Inflow',
        'Emergency_Cases',
        'Bed_Occupancy_Rate',
        'Surgery_Count',
        'Operating_Theater_Utilization',
        'Holiday_Flag'
    ]

    scaler = MinMaxScaler()
    df[state_columns] = scaler.fit_transform(df[state_columns])

    states = df[state_columns].values
    actions = df['Adjusted_Doctors'].values

    return states, actions
