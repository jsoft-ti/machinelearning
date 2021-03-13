from resources.scripts.functions import *
import pandas as pd


def getInvestorProfiles():
    conn = getConnection()
    mycursor = conn.cursor()
    #query = "SELECT * FROM landing_investorprofile;"
    query = "select  ip.id, ip.r1, ip.r2, ip.r3, ip.r4, ip.r5, ip.r6, ip.r7 , ip.r8 , ip.r9, ip.r10, ip.r11, ip.r12, ip.auth_user_id_id, ip.broker_id_id, ip.profiletype_id from auth_user u, landing_investorprofile ip where u.id = ip.auth_user_id_id"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    my_columns = ['id', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'auth_user_id_id',
                  'broker_id_id', 'profiletype_id']
    return  pd.DataFrame(myresult, columns=my_columns)


def getDataportifolioStock():
    conn = getConnection()
    mycursor = conn.cursor()
    query = "select ipo.id, ipo.auth_user_id_id, ipo.stock_radar_id_id, sr.name from landing_modelinvestmentportfolio ipo, landing_stockradar sr where sr.id = ipo.stock_radar_id_id "
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    my_columns = ['id', 'auth_user_id_id', 'stock_radar_id_id', 'name']
    return  pd.DataFrame(myresult, columns=my_columns)

def getStockRadar():
    conn = getConnection()
    mycursor = conn.cursor()
    query = "select sr.id, sr.name, sr.status, sr.description from  landing_stockradar sr"
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    my_columns = ['id', 'name', 'status', 'description']
    return  pd.DataFrame(myresult, columns=my_columns)

def getRecommendationStock(user_ids):
    conn = getConnection()
    mycursor = conn.cursor()
    query_in = ""
    for u in user_ids:
        query_in = f"{u},{query_in}"
    query_in = query_in[:-1]
    query = f"select ipo.stock_radar_id_id, sr.name from landing_modelinvestmentportfolio ipo, landing_stockradar sr where sr.id = ipo.stock_radar_id_id and ipo.auth_user_id_id in ({query_in})"
    print(query)
    mycursor.execute(query)
    myresult = mycursor.fetchall()
    my_columns = ['id', 'name']
    return pd.DataFrame(myresult, columns=my_columns)

df_stock_radar = getStockRadar()
df_investor_profile = getInvestorProfiles()
stock_to_columns = df_stock_radar['name']
df_stocks = pd.DataFrame(columns=stock_to_columns)
df_final = pd.concat([df_investor_profile,df_stocks], axis=1).fillna(0)
df_all_portifolios = getDataportifolioStock()
#print(df_all_portifolios)
#df_final.index = df_final['auth_user_id_id']
#print(df_all_portifolios)
for i in range(0, len(df_all_portifolios)):
    user  = df_all_portifolios.iloc[i]['auth_user_id_id']
    stock = df_all_portifolios.iloc[i]['name']
    for j in range(0, len(df_final)):
        user2 = df_final.iloc[j]['auth_user_id_id']
        if user == user2:
           # print(df_final.iloc[j][stock])
            #df_final = df_final.copy()
            df_final.loc[j][stock] = 1
            #print("___________________________")
            #print(df_final.iloc[j][stock])
            #print("=============================")
            #print(j, stock)

def getRecommendation(param_user_auth_id):
    df_corr = df_final.copy()
    df_corr.index = df_corr.auth_user_id_id
    df_corr = df_corr.drop(columns = ['id'])


    try:
        selected_user = df_corr.iloc[param_user_auth_id]
    except:
        return []
    print(selected_user.auth_user_id_id)
    corr = df_final.corrwith(selected_user, axis = 1).sort_values(ascending = False).head(20)
    user_looks_like_id = df_final.iloc[corr[1:len(corr)].index]['auth_user_id_id']
    recomend_stock = getRecommendationStock(user_looks_like_id)
    recomend_stock_count = recomend_stock.groupby(by='name').count()
    print(recomend_stock_count)

teste = getRecommendation(3)
print(teste)
'''
2    1.000000
1    0.888306
0    0.885650


2    1.000000
0    0.816581
1    0.774617
'''