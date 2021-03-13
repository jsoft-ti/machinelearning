from resources.scripts.libs import *

###Conexão à base de dados
def getConnection():
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="coindb1",
        port="8889"
    )
    return my_db

MY_FORMAT = '%Y-%m-%d'

###Classe para padronização dos objetos manipulados
class Sumary():
    def __init__(self, coin, date, opening, closing, lowest, highest, volume, quantity, amount, avg_price):
        self.coin = coin
        self.date = date
        self.opening = opening
        self.closing = closing
        self.lowest = lowest
        self.highest = highest
        self.volume = volume
        self.quantity = quantity
        self.amount = amount
        self.avg_price = avg_price


###Criação padronizada de tabelas
def create_table(coin_name):
    print()
    command = f"CREATE TABLE {coin_name}_day_sumary (id int(11) NOT NULL PRIMARY KEY AUTO_INCREMENT, coin varchar(10) NOT NULL, date date NOT NULL, opening double NOT NULL, closing double NOT NULL, lowest double NOT NULL, highest double NOT NULL, volume double NOT NULL, quantity double NOT NULL, amount double NOT NULL, avg_price double NOT NULL)"
    cnn = getConnection()
    cnn.cursor().execute(f"drop table if exists {coin_name}_day_sumary;")
    cnn.commit()
    cnn.cursor().execute(command)
    cnn.commit()


###Solicita uma lista de papéis ou índices com base em uma data inicial até os dias de hoje
def getStock(stock_array, date_init):
    bag_amount = 20
    bag_control = 0
    date_now = dt.date.today()
    for stock in stock_array:
        bag_control = 0
        cnn = getConnection()
        table_name = stock.replace('.','')
        create_table(table_name)
        tickerData = yf.Ticker(stock)
        tickerDf = pd.DataFrame(
            tickerData.history(start=date_init.strftime("%Y-%m-%d"), end=date_now.strftime("%Y-%m-%d")))

        for index, stk in tickerDf.iterrows():

            try:
                command = f"INSERT INTO {table_name}_day_sumary (coin, date, opening, closing, lowest, highest, volume, quantity, amount, avg_price) values ('{stock}', '{index}', '{stk.Open}', '{stk.Close}', '{stk.Low}', '{stk.High}', '{stk.Volume}', '0', '0', '0')"

                cnn.cursor().execute(command)
                if (bag_control >= bag_amount):
                    cnn.commit()
                    bag_control = 0
                    # print(calc_date)
            except Exception as e:
                print(e)
                continue
        cnn.commit()
    cnn.commit()
    print("Finalizando aquisição de dados " + stock)


# Solicita o histórico de negociações de uma lista de cryptomoedas com base em uma data inicial
def getCripto(coin_array, date_init):
    print("Dentro do Cripto")
    cnn = getConnection()
    date_now = dt.date.today()
    print("Calculando a data")
    for coin in coin_array:
        create_table(coin)
        calc_date = date_init
        bag_amount = 20
        bag_control = 0
        # Descomentar se desejar excluir todos os dados anteriores
        # command = f"delete from {coin}_day_sumary"
        # cnn.cursor().execute(command)
        # cnn.commit()
        while calc_date.date() < date_now:
            bag_control += 1
            calc_date += dt.timedelta(days=1)
            if (calc_date.weekday() > 5):
                continue
            sumary_json = DataAPI.day_summary(coin, calc_date.year, calc_date.month, calc_date.day).json()
            sumary_json["coin"] = coin
            try:
                sumary = Sumary(**sumary_json)
                command = f"INSERT INTO {coin}_day_sumary (coin, date, opening, closing, lowest, highest, volume, quantity, amount, avg_price) values ('{sumary.coin}', '{sumary.date}', '{sumary.opening}', '{sumary.closing}', '{sumary.lowest}', '{sumary.highest}', '{sumary.volume}', '{sumary.quantity}', '{sumary.amount}', '{sumary.avg_price}')"
                # print(command)
                cnn.cursor().execute(command)
                if (bag_control >= bag_amount):
                    cnn.commit()
                    bag_control = 0
            except Exception as e:
                cnn.commit()
                bag_control = 0
                # print(f"Deu erro -> {calc_date} {e}")

        print("")
        cnn.commit()



# Carrega os dados de uma moeda já existente no banco
def getData(coin):
    conn = getConnection()
    mycursor = conn.cursor()
    query = f"SELECT * FROM {coin}_day_sumary;"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    return myresult


# Carrega os dados do BTC e Indices diretamente do banco
def get_data_from_db():
    dbColumns = 'btcopening,btcclosing,btcquantity,btcamount,daxopening,daxclosing,daxlowest,daxhighest,djiopening,djiclosing,djilowest,djihighest,hsiopening,hsiclosing,hsilowest,hsihighest,date'.split(
        ',')
    # dbColumns = 'btcopening,btcclosing,btclowest,btchighest,btcvolume,btcquantity,btcamount,btcavgprice,daxopening,daxclosing,daxlowest,daxhighest,djiopening,djiclosing,djilowest,djihighest,hsiopening,hsiclosing,hsilowest,hsihighest,date'.split(',')
    query = open('loaddata.sql').read()
    conn = getConnection()
    mycursor = conn.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    return pd.DataFrame(myresult, columns=dbColumns)


# EDA Lúcio/Edu
def normal(df, col, threshold=0.05):
    try:
        zscore, p_value = stats.normaltest(df[col])
        if p_value < threshold:
            result = 'not_normal'
        else:
            result = 'normal'
    except:
        zscore = p_value = np.nan
        result = 'not_applicable'
    return result


def outliers_count_IQR(df, col):
    try:
        if len(df[col].unique()) > 2:  # if para eliminar features binárias
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr_range = q3 - q1
            lower = q1 - 1.5 * iqr_range
            upper = q3 + 1.5 * iqr_range
            out_low = df[df[col] < lower].count()[0]
            out_up = df[df[col] > upper].count()[0]
            outliers = out_low + out_up
            outliers_perc = round(outliers / df.shape[0], 2)
        else:
            outliers = np.nan
            outliers_perc = np.nan
    except:
        outliers = np.nan
        outliers_perc = np.nan
    return outliers, outliers_perc


# o que interessa
def EDA_LucioEdu(df):
    df = df.rename(columns=str.lower)

    eda_df = {}
    eda_df['Amount_NaN'] = df.isnull().sum()
    eda_df['%_NaN'] = df.isnull().mean().round(2)
    eda_df['DType'] = df.dtypes
    eda_df['Amount_Data'] = df.count()

    colunas = df.columns.tolist()

    eda_df['Amount_Unique'] = pd.Series(map(lambda x: len(df[x].unique().tolist()), colunas), index=colunas)

    eda_df['Min'] = df.min()
    eda_df['Max'] = df.max()
    eda_df['Mean'] = df.mean().round(3)
    eda_df['STD'] = df.std().round(3)

    eda_df['Normality'] = pd.Series(map(lambda x: normal(df, x), colunas), index=colunas)
    eda_df['Amount_Outliers'] = pd.Series(map(lambda x: outliers_count_IQR(df, x)[0], colunas), index=colunas)
    eda_df['%_Outliers'] = pd.Series(map(lambda x: outliers_count_IQR(df, x)[1], colunas), index=colunas)
    df = pd.DataFrame(eda_df)
    return df.loc[colunas, :]


# Plotagem padrão do histórico do papel
def plot_stock_price_default(df):
    df = df.reset_index()
    stock = ColumnDataSource(
        data=dict(opening=[], closing=[], highest=[], lowest=[], index=[]))
    stock.data = stock.from_df(df)

    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Stock price", toolbar_location='above')

    inc = stock.data['closing'] > stock.data['opening']
    dec = stock.data['opening'] > stock.data['closing']
    view_inc = CDSView(source=stock, filters=[BooleanFilter(inc)])
    view_dec = CDSView(source=stock, filters=[BooleanFilter(dec)])

    p.segment(x0='index', x1='index', y0='lowest', y1='highest', color=RED, source=stock, view=view_inc)
    p.segment(x0='index', x1='index', y0='lowest', y1='highest', color=GREEN, source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='opening', bottom='closing', fill_color=BLUE, line_color=BLUE,
           source=stock, view=view_inc, name="avg_price")
    p.vbar(x='index', width=VBAR_WIDTH, top='opening', bottom='closing', fill_color=RED, line_color=RED,
           source=stock, view=view_dec, name="avg_price")

    # p.legend.location = "top_left"
    # p.legend.border_line_alpha = 0
    # p.legend.background_fill_alpha = 0
    # p.legend.click_policy = "mute"
    elements = list()
    elements.append(p)

    curdoc().add_root(column(elements))
    curdoc().title = 'Bokeh stocks historical prices'
    show(p, notebook_handle=True)


def plot_stock_closing(df_closing):
    plt.figure(figsize=(22, 10))
    plt.plot(df_closing['btc'], label='BitCoin')
    plt.plot(df_closing['dji'], label='DowJones')
    plt.plot(df_closing['dax'], label='DAX-Alemanha')
    # plt.plot(df_closing['hsi'], label='HSI - HongKong')
    plt.legend(loc="upper left", fontsize='large')
    plt.title("Closing Series")
    plt.ylabel("Closing")
    plt.show()


# Imprime matriz de correlação
def get_correlation(df):
    corrmat = df.corr()
    sns.set(font_scale=1.0)
    f, ax = plt.subplots(figsize=(15, 10))
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrmat, dtype=bool))
    hm = sns.heatmap(corrmat,
                     mask=mask,
                     cmap='icefire',
                     cbar=True,  # formatando a barra lateral de cores para o heatmap
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=corrmat.columns,
                     xticklabels=corrmat.columns)


def break_date(df, column):
    df['year'] = pd.to_datetime(df[column], format=MY_FORMAT).dt.year
    df['month'] = pd.to_datetime(df[column], format=MY_FORMAT).dt.month
    df['day'] = pd.to_datetime(df[column], format=MY_FORMAT).dt.day
    return df.drop(columns=[column])


# MAPE

def MAPE(y_real, y_pred):
    mape_list = []
    for i in range(0, len(y_real)):
        mape_list.append(np.abs((y_real[i] - y_pred[i]) / y_real[i]))

    return (np.mean(mape_list))

def setupData():
    ### Campos do objeto Summary
    data_columns = ['id', 'coin', 'date', 'opening', 'closing', 'lowest', 'highest', 'volume', 'quantity', 'amount',
                    'avg_price'];

    ### Intervalo de datas
    date_init = dt.datetime(2015, 1, 1, 0, 0, 0)


    date_now = dt.date.today()

    ### Definindo quais Criptomoedas e quais índices serão utilizados
    coin_array = ['BTC']
    #stock_array = ['DJI', 'DAX', 'HSI', 'PETR4.SA', 'ABEV3', 'CMIG4', 'GGBR4', 'ITUB4', 'BBDC4', 'BBAS3', 'VALE3']
    stock_array = ['PETR4.SA']

    # 1 - Este processo carrega os dados utilizados na análise por meio das APIs Mercado Bitcoin
    # e Yahoo Fynance¶
    #getCripto(coin_array, date_init)
    getStock(stock_array, date_init)

def generatePickleNextBTC(data):
    with open('resources/scripts/main_variables.yaml') as f:
        df = get_data_from_db()
        #df = pd.read_csv(data['data_path'], sep=',', encoding='utf-8')

        df2 = df.drop(columns=data['columns_to_remove'])
        df2 = break_date(df2, 'date')
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
            df_result = pd.DataFrame(columns=['date', 'value'])
            df_result['value'] = pred_teste.round(3)
            df_result['date'] = X.index[test_index]
            # print(df_result)
            # display()
            # y_treino = y_treino.reset_index().drop(columns = 'date').to_numpy()
            metrica_treino.append(MAPE(y_treino, pred_treino))

            metrica_teste.append(MAPE(y_teste, pred_teste))

            # diff_metrica.append(100*np.abs((MAPE(y_treino, pred_treino) - \
            # MAPE(y_teste, pred_teste))/\
            # MAPE(y_teste, pred_teste)))
        df_r = pd.DataFrame({'Modelo': "LGBM", 'Periodo': data['max_train_size'], \
                             'Treino': np.mean(metrica_treino), 'Teste': np.mean(metrica_teste)}, index=[0])

        # print(np.mean(metrica_teste))
        # print(np.mean(metrica_treino))
        # metrica["Final_teste_"+model] = metrica_final_teste
        # metrica["Final_treino_"+model] = metrica_final_treino
        # metrica['diff'] = (metrica['Teste'] - metrica['Treino']).round(2)
        diff = np.mean(metrica_teste) - np.mean(metrica_treino)
        # print(diff.round(2))
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
