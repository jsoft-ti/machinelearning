#######Manipulação de dados e operações matemáticas
import pandas as pd
import numpy as np
import array

#######Padronização de respostas rest
import json
import mysql.connector

#######Api de consulta de CriptoMoedas e Índices
import yfinance as yf
from mercado_bitcoin import DataAPI

#######Conversão de dados temporais
import datetime as dt

#Visualização de dados
import seaborn as sns
import matplotlib.pyplot as plt
import os
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.widgets import Dropdown
from bokeh.io import curdoc, push_notebook, show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import BooleanFilter, CDSView, Select, Range1d, HoverTool
from bokeh.palettes import Category20
from bokeh.models.formatters import NumeralTickFormatter

#Pre-processamento
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


#Otimização de Modelos
from sklearn.model_selection import GridSearchCV


#Modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

#Explicação do modelo
import shap


#Avaliação de Modelos
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

#Salvar o modelo para produção
import joblib