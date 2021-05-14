# -*- coding: utf-8 -*-
"""
Statistic 
não esquecer de:
    set console working directory


"""

import pandas as pd
Data_Set1 = pd.read_csv('Data_Set.csv')

Data_Set2 = pd.read_csv('Data_Set.csv', header=2)
#Mudando o index Temperature para Temp
Data_Set3 = Data_Set2.rename(columns = {'Temperature': 'Temp'})

Data_Set4= Data_Set3.drop('No. Occupants',axis=1)

#Implace faz a mudança no Data_Set3
Data_Set3.drop('No. Occupants',axis=1, inplace=True)
#Removendo a linha 2 do index
Data_Set5 = Data_Set4.drop(2, axis = 0)
#Resetar o index para não pular o index do 1 para o 3
Data_Set6 = Data_Set5.reset_index(drop=True)
#.scribe mostra muitas informações importantes como média e desvio padrão
Data_Set6.describe()
#Encontrar o menor valor em E_Heat
Min_item = Data_Set6['E_Heat'].min()

Data_Set6['E_Heat'][Data_Set6['E_Heat']==Min_item]
#Trocar o valor -4 pelo 21
Data_Set6['E_Heat'].replace(-4,21, inplace=True)
#Covariancia
Data_Set6.cov()

import seaborn as sn

sn.heatmap(Data_Set6.corr())

#Missing values

Data_Set6.info()

import numpy as np


Data_Set7 = Data_Set6.replace('!', np.NaN)

Data_Set7.info()

Data_Set7 = Data_Set7.apply(pd.to_numeric)
#Mostra os valores null
Data_Set7.isnull()

Data_Set7.drop(13, axis=0, inplace=True)
#Deleta os valores nan
Data_Set7.dropna(axis=0, inplace=True)
#Trocar os valores nan: pode ser ffill ou bfill
Data_Set8 = Data_Set7.fillna(method = 'bfill')

#Trocar os valores nan pela média
from sklearn.impute import SimpleImputer

M_Var = SimpleImputer(missing_values=np.nan, strategy='mean')

M_Var.fit(Data_Set7)

Data_Set9 = M_Var.transform(Data_Set7)

""""Outlier Detection"""

Data_Set8.boxplot()

Data_Set8['E_Plug'].quantile(0.25)
Data_Set8['E_Plug'].quantile(0.75)
"""
Q1 = 19.75 
Q3 = 32.25
IQ = 32.25-19.75 = 12.5

#Mild Outlier

Lower Bound = Q1-1.5*IQR = 19.75 - 1.5*12.5 = 1
Upper Bound = Q3+1.5*IQR = 32.25 + 1.5*12.5 = 51

#Extreme Outlier 

Upper Bound = Q3+3*IQR = 32.25+3*12.5 = 69.75
"""

Data_Set8['E_Plug'].replace(120, 42, inplace=True)


#Concatenation
#Juntar Index em linhas ou colunas
New_Col = pd.read_csv('Data_New.csv')

Data_Set10 = pd.concat([Data_Set8, New_Col], axis=1)

"""Dummy Variables"""
Data_Set10.info()

Data_Set11 = pd.get_dummies(Data_Set10)

Data_Set11.info()

"""Normalizaton"""

from sklearn.preprocessing import minmax_scale, normalize 

#First Method: Min Max Scale

Data_Set12 = minmax_scale(Data_Set11, feature_range=(0,1))

Data_Set13 = normalize(Data_Set11, norm = '12',axis=0)
#axis=0 is for normalizing features; axis =1 is for normalizing sample

Data_Set13 = pd.DataFrame(Data_Set13, columns=['Time', 'E_Plug', 'E_Head', 'Price', 'Temp', 'OffPeak', 'Peak'])