# Persiapkan Package yang akan digunakan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# membuka dan membaca file data tangkapan ikan
data_tangkapan = pd.read_csv('Hasil tangkapan & variabel lingkungan.csv', header=0, index_col=0)
data_tangkapan.set_index('datetime', inplace=True)  # Setting 'datetime' as index
#data_tangkapan.drop(data_tangkapan.columns[0], axis=1, inplace=True)

# Menentukan kolom yang akan menjadi variabel prediktor dan variabel target
predictor_cols = ['sst_dailymean (C)','chlorofil_dailymean (mg/m^3)','northward_wind_dailymean (m/s)','eastward_wind_dailymean (m/s)','northward_seawater_dailymean (m/s)', 'eastward_seawater_dailymean (m/s)','seawater_magnitude_dailymean (m/s)','wind_magnitude_dailymean (m/s)']
predictor = data_tangkapan[predictor_cols]

#target_cols = ['Maesang (Cakalang)']
target = data_tangkapan['Maesang (Cakalang)']

# Membagi dataset untuk TRAIN dan TEST
from sklearn.model_selection import train_test_split
predict_train, predict_test, target_train, target_test = train_test_split(predictor, target, train_size=0.4, random_state = 42)

# Melatih model regresi linear dengan data training
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(predict_train, target_train)

# make pickle file of our model
pickle.dump(lin_reg, open("LinReg model.pkl", "wb"))