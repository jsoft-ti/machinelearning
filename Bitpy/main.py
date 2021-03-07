from resources.scripts.libs import *
from resources.scripts.functions import *
from resources.scripts.predictor import *
import pickle
from datetime import datetime
import yaml


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parameters
    with open('resources/scripts/main_variables.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        df = pd.read_csv(data['data_path'], sep=',', encoding='utf-8')

        df2 = df.drop(columns=data['columns_to_remove'])

        df_closing = pd.DataFrame(columns=data['df_closing_columns'])

        df_closing['btc'] = df2['btcclosing']
        df_closing['dji'] = df2['djiclosing']
        df_closing['dax'] = df2['daxclosing']
        df_closing['hsi'] = df2['hsiclosing']
        df_closing['year'] = df2['year']
        df_closing['month'] = df2['month']
        df_closing['day'] = df2['day']

        df_closing_1 = df_closing.drop(columns='hsi')  # vou dropar essa feature porque está com valores muito estranhos
        df_closing_1['day'] = df2['day'].astype(str)
        df_closing_1['month'] = df_closing['month'].astype(str)
        df_closing_1['year'] = df_closing['year'].astype(str)
        df_closing_1['data'] = df_closing_1[['day', 'month', 'year']].agg('-'.join, axis=1)
        df_closing_1['day'] = df2['day'].astype(int)
        df_closing_1['month'] = df_closing['month'].astype(int)
        df_closing_1['year'] = df_closing['year'].astype(int)
        df_closing_1['date'] = pd.to_datetime(df_closing_1['data'], format='%d-%m-%Y')
        df_closing_1['day_of_year'] = df_closing_1.date.apply(lambda x: x.dayofyear)
        df_closing_1['week_of_year'] = df_closing_1.date.apply(lambda x: x.weekofyear)
        df_closing_1['day'] = df2['day'].astype(int)
        df_closing_1['initial_month'] = np.where(df_closing_1['day'] < 5, 1, 0)
        del df_closing_1['data']

        window = data['window']  # janela semanal, 5 dias úteis
        df_closing_2 = df_closing_1.copy()
        df_closing_2['week_median_btc'] = df_closing_1['btc'].rolling(window=window).mean().round(2)
        df_closing_2['week_median_dji'] = df_closing_1['dji'].rolling(window=window).mean().round(2)
        df_closing_2['week_median_dax'] = df_closing_1['dax'].rolling(window=window).mean().round(2)

        df_closing_3 = df_closing_2.copy()
        df_closing_3['variacao_sem_btc'] = df_closing_3['btc'].pct_change(window - 1).round(4)
        df_closing_3['variacao_sem_dji'] = df_closing_3['dji'].pct_change(window - 1).round(4)
        df_closing_3['variacao_sem_dax'] = df_closing_3['dax'].pct_change(window - 1).round(4)
        # correlação entre btc e dji e btc e dax
        df_closing_3['corr_btc_dji'] = df_closing_3['btc'].rolling(window).corr(df_closing_3['dji'])
        df_closing_3['corr_btc_dax'] = df_closing_3['btc'].rolling(window).corr(df_closing_3['dax'])

        df_closing_4 = df_closing_3.copy()
        df_closing_4['dji'].shift(periods=window)
        df_closing_4['dax'].shift(periods=window)
        df_closing_4['week_median_btc'].shift(periods=window)
        df_closing_4['week_median_dji'].shift(periods=window)
        df_closing_4['week_median_dax'].shift(periods=window)
        df_closing_4['variacao_sem_btc'].shift(periods=window)
        df_closing_4['variacao_sem_dji'].shift(periods=window)
        df_closing_4['variacao_sem_dax'].shift(periods=window)
        df_closing_4['corr_btc_dji'].shift(periods=window)
        df_closing_4['corr_btc_dax'].shift(periods=window)
        df_closing_4 = df_closing_4.dropna()

        df_closing_5 = df_closing_4.copy()
        df_closing_5.index = df_closing_5.date
        df_closing_5 = df_closing_5.drop(columns=['date'])
        df_closing_5.index = df_closing_5.index.to_period('D')
        X = df_closing_5.drop(columns=['btc'])
        Y = df_closing_5.btc

        lgbm = lgb.LGBMRegressor(max_depth=data['max_depth'], num_leaves=data['num_leaves'],
                                 n_estimators=data['n_estimators'])
        tsp = TimeSeriesSplit(gap=data['gap'], max_train_size=data['max_train_size'], n_splits=data['n_splits'],
                              test_size=data['test_size'])
        metrica_teste = []
        metrica_treino = []
        diff_metrica = []
        for train_index, test_index in tsp.split(X.index):
            x_treino, x_teste = X.iloc[train_index], X.iloc[test_index]
            y_treino, y_teste = Y[train_index], Y[test_index]
            scaler = StandardScaler().fit(x_treino)
            x_treino_norm = scaler.fit_transform(x_treino)
            x_teste_norm = scaler.fit_transform(x_teste)

            my_model = lgbm.fit(x_treino_norm, y_treino)

            pred_treino = my_model.predict(x_treino_norm)
            pred_teste = my_model.predict(x_teste_norm)
            df_result = pd.DataFrame(columns = ['date', 'value'])
            df_result['value'] = pred_teste.round(3)
            df_result['date'] = X.index[test_index]
            #print(df_result)
            # display()
            # y_treino = y_treino.reset_index().drop(columns = 'date').to_numpy()
            metrica_treino.append(MAPE(y_treino, pred_treino))

            metrica_teste.append(MAPE(y_teste, pred_teste))

            # diff_metrica.append(100*np.abs((MAPE(y_treino, pred_treino) - \
            # MAPE(y_teste, pred_teste))/\
            # MAPE(y_teste, pred_teste)))
        df_r = pd.DataFrame({'Modelo': "LGBM", 'Periodo': data['max_train_size'], \
                             'Treino': np.mean(metrica_treino), 'Teste': np.mean(metrica_teste)}, index=[0])

        #print(np.mean(metrica_teste))
        #print(np.mean(metrica_treino))
        # metrica["Final_teste_"+model] = metrica_final_teste
        # metrica["Final_treino_"+model] = metrica_final_treino
        # metrica['diff'] = (metrica['Teste'] - metrica['Treino']).round(2)
        diff = np.mean(metrica_teste) - np.mean(metrica_treino)
        #print(diff.round(2))
        model = predictor(scaler, my_model)
        model.min_date = min(X.index[test_index])
        model.max_date = min(X.index[test_index])
        model.mape_train = np.mean(metrica_teste)
        model.mape_test = np.mean(metrica_teste)
        model.feature_importance = my_model.feature_importances_
        model.train_size = data['max_train_size']
        model.test_size = data['test_size']
        model.train_date = datetime.now()
        print(model)

        with open('output/serialized_model.pkl', 'wb') as pickle_file:
            pickle.dump(model, pickle_file)
        pickle_file.close()