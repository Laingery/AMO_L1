import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def data_preprocessing(df, scaler = None):

    # start_date = pd.Timestamp('2024-11-01')
    start_date=0

    df['full_datetime'] = (
        # start_date +
        pd.to_timedelta(df['Номер_дня'], unit='D') +
        pd.to_timedelta(df["Время_дня"]
                        + ':00'
                        )
    )

    df['minutes_from_day'] = pd.to_datetime(df["Время_дня"], format='%H:%M').dt.hour * 60\
                                  + pd.to_datetime(df["Время_дня"], format='%H:%M').dt.minute

    df['minutes_sin'] = np.sin(2 * np.pi * df['minutes_from_day'].values / (24.0*60))
    df['minutes_cos'] = np.cos(2 * np.pi * df['minutes_from_day'].values / (24.0*60))

    df = df.drop(['Номер_дня', 'Время_дня'], axis=1)

    if scaler == None:
        scaler = StandardScaler().fit(df[['Температура_A1_°C', 'Температура_A2_°C']])
        joblib.dump(scaler, './scaler.bin')

    df_Xtransform = scaler.transform(df[['Температура_A1_°C', 'Температура_A2_°C']])

    #
    # print(df.head(10))
    df[['t_A1Norm', 't_A2Norm']] = df_Xtransform

    X = df[['minutes_from_day', 'minutes_sin', 'minutes_cos', 't_A1Norm', 't_A2Norm']]
    Y = df['Температура_A3_°C']

    # X = df[['minutes_from_day', 'minutes_sin', 'minutes_cos', 't_A1Norm', 't_A2Norm']]
    # Y = df['t_A3Norm']

    return X, Y, scaler

scaler = None

X_train, Y_train, scaler = data_preprocessing(pd.read_csv('train/train.csv', sep=';').sort_values(by='Номер_дня'))
X_train.to_csv("train/train_features.csv", index=False, encoding='utf-8-sig', sep=';')
Y_train.to_csv("train/train_labels.csv", index=False, encoding='utf-8-sig', sep=';')
#
X_test, Y_test, scaler = data_preprocessing(pd.read_csv('test/test.csv', sep=';').sort_values(by='Номер_дня')
                                    , scaler)
X_test.to_csv("test/test_features.csv", index=False, encoding='utf-8-sig', sep=';')
Y_test.to_csv("test/test_labels.csv", index=False, encoding='utf-8-sig', sep=';')



