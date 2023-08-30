import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

warnings.filterwarnings('ignore')
indexState = 2167
class model:
    dataframe = None
    dataframe_UI = None
    case = None
    Via = None
    Citta = None
    Regione = None
    Anno_c = None
    Anno_r = None
    Living = None
    Lot= None
    Piani = None
    WaterFront= None
    Basement= None
    Stanze = None
    Vista = None
    Condizione = None
    Above= None
    Modello = None

    streets = None
    citys = None
    countrys = None

    features = ['sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated','street','city','country','rooms']

# Function to map house prices to price range categories
def map_to_price_range(price):
    price_ranges = [
        (0, 80000),
        (80000, 150000),
        (150000, 200000),
        (200000, 650000),
        (650000, 1000000),
        (1000000, 3000000),
        (3000000, float('inf'))
    ]
    for idx, (lower, upper) in enumerate(price_ranges):
        if lower <= price < upper:
            return idx
    return len(price_ranges)

def crea_basedati(modelUsed='None'):
    #read data
    """
    # Leggi il file CSV
    df = pd.read_csv("data.csv")

    # Somma le colonne "bedrooms" e "bathrooms" e inserisci il risultato nella nuova colonna "total_rooms"
    df["rooms"] = df["bedrooms"] + df["bathrooms"]

    # Elimina le colonne "bedrooms" e "bathrooms"
    df.drop(["bedrooms", "bathrooms"], axis=1, inplace=True)

    # Salva il file CSV con la nuova colonna "total_rooms"
    df.to_csv("house.csv", index=False)

    """
    #pesi delle features 
    """
    # Carica i dati in un DataFrame
    data = pd.read_csv('house.csv')

    # Seleziona le features (colonne) che si desidera utilizzare per l'addestramento del modello
    X = data[[ 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated','rooms']]

    # Converti le colonne contenenti stringhe in numeri utilizzando l'encoding one-hot
    X = pd.concat([X, pd.get_dummies(data[['street', 'city', 'country']])], axis=1)

    # Seleziona la colonna degli output (variabile dipendente)
    y = data['price']

    # Crea un modello di regressione ad albero decisionale
    model = DecisionTreeRegressor()

    # Addestra il modello sul set di dati
    model.fit(X, y)

    # Ottieni l'importanza delle features dal modello
    importances = model.feature_importances_

    # Stampa l'importanza delle features
    for i in range(len(X.columns)):
        print(X.columns[i], ':', importances[i])
    """

    house_data = pd.read_csv("house.csv", index_col=False)
    
    data = pd.DataFrame(house_data)
    #Conversione dei tipi dei dati 
    data['price']     = data['price'].astype('int64')
    data['rooms']     = data['rooms'].astype('float32')
    data['floors']    = data['floors'].astype('float32')
    data['street']    = data['street'].astype('string')
    data['city']      = data['city'].astype('string')
    data['statezip']  = data['statezip'].astype('string')
    data['country']   = data['country'].astype('string')

    data.drop_duplicates(inplace=True)

    data['price'].replace(0, np.nan, inplace = True)
    data.isnull().sum()
    data.dropna(inplace=True)

    indexDf = len(data)
    #Conversione piedi quadri in metri quadri
    scaleFeet = 10.764

    for i in tqdm(range(indexDf)):
        data.iloc[i, 2] = float(data.iloc[i, 2]) / scaleFeet # 2 = sqft_living
        data.iloc[i, 3] = float(data.iloc[i, 3]) / scaleFeet # 3 = sqft_lot
        data.iloc[i, 8] = float(data.iloc[i, 8]) / scaleFeet # 8 = sqft_above
        data.iloc[i, 9] = float(data.iloc[i, 9]) / scaleFeet # 9 = sqft_basement

    case = data.copy()
    model.dataframe_UI = case 
    model.Citta = sorted(list(case.city.unique()))
    model.Regione = sorted(list(case.country.unique()))
    model.Via = sorted(list(case.street.unique()))
    model.Anno_c = sorted(list(case.yr_built.unique()))
    model.Anno_c.reverse()
    model.Anno_r = sorted(list(case.yr_renovated.unique()))
    model.Anno_r.reverse()
    model.Living =sorted(list(case.sqft_living.unique()))
    model.Lot = sorted(list(case.sqft_lot.unique()))
    model.Piani = sorted(list(case.floors.unique()))
    model.WaterFront = sorted(list(case.waterfront.unique()))
    model.Basement = sorted(list(case.sqft_basement.unique()))
    model.Stanze = sorted(list(case.rooms.unique()))
    model.Above = sorted(list(case.sqft_above.unique()))
    model.Vista = sorted(list(case.view.unique()))
    model.Condizione = sorted(list(case.condition.unique()))
    model.Modello = ['Random Forest', "SGD"]
    
    house = pd.get_dummies(data, columns=['city'], prefix=['city'])

    label_encoder = LabelEncoder()
    house['street'] = label_encoder.fit_transform(house['street'])
    house['country'] = label_encoder.fit_transform(house['country'])

    house = house.drop(['date', 'statezip'], axis = 1)
    columns = house.columns
    columns = columns.drop('price')
    
    # Processo per rendere i dati degli scalari 
    scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
    
    normal = pd.DataFrame(scaler.fit_transform(house.loc[:, house.columns!='price']), columns = columns)
    normal = normal.reset_index(drop=True, inplace=False)

    model.dataframe = normal.copy()

    prices_x = normal
    # ---------------- #
    if(modelUsed == 'RandomForest'):
        prices_y = pd.DataFrame(house["price"]) # Prezzi normali
    elif (modelUsed == 'SGD'):
        prices_y = pd.DataFrame(house["price"])
        # Applico la funzione per creare la variabile target (categorie fascia di prezzo)
        prices_y['price'] = prices_y['price'].apply(map_to_price_range)
    else:
        raise NotImplementedError(F"Parametro modelUsed non corretto. Hai usato: '{modelUsed}'.")  

    # Apprendimento non supervisionato con clustering
    clusters = DBSCAN(eps=0.9, min_samples=3).fit(prices_x)
    prices_x["noise"] = clusters.labels_
    prices_y["noise"] = clusters.labels_
    prices_x = prices_x[prices_x.noise>-1]
    prices_y = prices_y[prices_y.noise>-1]
    prices_x.drop('noise', inplace = True, axis=1)
    prices_y.drop('noise', inplace = True, axis=1)

    # Train/test split    
    np.random.seed(indexState)
    prices_x_train, prices_x_test, prices_y_train, prices_y_test = train_test_split(prices_x, prices_y, test_size=0.2)
    
    prices_x_train = prices_x_train.to_numpy()
    prices_x_test = prices_x_test.to_numpy()
    prices_y_train = prices_y_train.to_numpy()
    prices_y_test = prices_y_test.to_numpy()
    
    # print('Training set size: %d' %len(prices_x_train))
    # print('Test set size: %d' %len(prices_x_test))
    # print('----------------------------------------------')
    # print(F'Shape of X values for Training set: {prices_x_train.shape}')
    # print(F'Shape of Y values for Training set: {prices_y_train.shape}')
    # print('----------------------------------------------')
   
    return prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler

#Funzione per plottare i grafici
def plot(predictions, x, y, n_feature):
    fig = plt.figure(dpi=125)
    fig.set_figwidth(10)

    ax = fig.add_subplot(1, 2, 1, projection='3d')        
    ax.elev = 20
    ax.azim = 20
    ax.scatter3D(x[:,n_feature], y, edgecolors='blue', alpha=0.5)
    ax.scatter3D(x[:,n_feature], predictions, 0.00, linewidth=0.5, edgecolors='red', alpha=0.7)
    
    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.elev = 20
    ax.azim = 20

    ax.scatter3D(x[:,n_feature], y, edgecolors='blue', alpha=0.5)
    ax.scatter3D(x[:,n_feature], predictions, 0.02, linewidth=0.5, edgecolors='red', alpha=0.7)

    title = ""
        
    plt.suptitle(title)            
    plt.show()

def modello2(prices_x_train, prices_x_test, prices_y_train, prices_y_test):

    # Create and train the SGD model
    SGD_model = SGDClassifier(loss="log", n_jobs=-1, alpha=0.0001, random_state=indexState)

    # # ========== Grid Search  ========== #
    # xConcatenato = np.concatenate((prices_x_train, prices_x_test), axis=0)
    # yConcatenato = np.concatenate((prices_y_train, prices_y_test), axis=0)
    #
    # cv = KFold(n_splits=10, shuffle=True, random_state=indexState)
    # listaAccuracy = []
    # i = 0
    # print("-- Grid Search per SGD --\n")

    # for tr_idx, test_idx in cv.split(xConcatenato, yConcatenato):
    #     i += 1
    #     correctGrid = 0
    #     X_train, X_test = xConcatenato[tr_idx], xConcatenato[test_idx]
    #     y_train, y_test = yConcatenato[tr_idx], yConcatenato[test_idx]

    #     grid_search_regression = GridSearchCV(SGD_model,
    #                             {
    #                             'alpha':np.arange(0.0001,0.0006,0.0001),                            
    #                             'fit_intercept': [True, False],
    #                             }, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1
    #                             )

    #     grid_search_regression.fit(X_train, np.squeeze(y_train).ravel())

    #     gridSearchPred = grid_search_regression.predict_proba(X_test)
    #     gridSearchPred = np.argmax(gridSearchPred, axis=1)
    #     correctGrid += (gridSearchPred == np.squeeze(y_test)).sum()
    #     accuracyPerFold = (correctGrid/y_test.size)*100
    #     listaAccuracy.append(accuracyPerFold)
    #     print(F"Split n.{i} | Best hyper-parameters per SGD: {grid_search_regression.best_params_}")
    #     print(f"Accuracy split n.{i}: {accuracyPerFold:.2f}%")

    # split_numbers = np.arange(1, 10 + 1)
    # plt.plot(split_numbers, listaAccuracy, label='Accuracies')
    # plt.xlabel('Split')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Grid Search - SGD')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # ========== SGD normale ========== #
    correct = 0
    SGD_model.fit(prices_x_train, prices_y_train)
    y_pred = SGD_model.predict_proba(prices_x_test)
    y_pred_probabilities = y_pred
    y_pred = np.argmax(y_pred, axis=1)
    correct += (y_pred == np.squeeze(prices_y_test)).sum()

    print(f"Accuracy SGD: {(correct/prices_y_test.size)*100:.2f}%")

    disp = printConfusionMatrix(prices_y_test, y_pred, SGD_model, 'SGD').plot()

    return SGD_model

def modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test):

    # Apprendimento supervisionato
    forest_model = RandomForestRegressor(n_jobs=-1, random_state=indexState)

    # # ========== Grid Search ========== #
    # xConcatenato = np.concatenate((prices_x_train, prices_x_test), axis=0)
    # yConcatenato = np.concatenate((prices_y_train, prices_y_test), axis=0)

    # cv = KFold(n_splits=10, shuffle=True, random_state=indexState)
    # listaR2Scores = []
    # i = 0
    # print("-- Grid Search per forestmodel --\n")
    # for tr_idx, test_idx in cv.split(xConcatenato, yConcatenato):
    #     i += 1
    #     X_train, X_test = xConcatenato[tr_idx], xConcatenato[test_idx]
    #     y_train, y_test = yConcatenato[tr_idx], yConcatenato[test_idx]

    #     grid_search_regression = GridSearchCV(forest_model,
    #                             {
    #                             'n_estimators':np.arange(100,200,10),                            
    #                             'criterion': ['mae', 'friedman_mse'],
    #                             'bootstrap': [True, False],
    #                             'warm_start': [True, False],
    #                             }, cv=cv, scoring="neg_root_mean_squared_error", verbose=10, n_jobs=-1
    #                             )

    #     grid_search_regression.fit(X_train, np.squeeze(y_train).ravel())

    #     gridSearchPred = grid_search_regression.predict(X_test)
    #     r2score = r2_score(y_test, gridSearchPred)

    #     listaR2Scores.append(r2score)
    #     print(F"Split n.{i} | Best hyper-parameters per Forest Model: {grid_search_regression.best_params_}")
    #     print(f"R2 score split n.{i}: {r2score:.2f}%")

    # split_numbers = np.arange(1, 10 + 1)
    # plt.plot(split_numbers, listaR2Scores, label='Scores')
    # plt.xlabel('Split')
    # plt.ylabel('R2 Score')
    # plt.title('R2 Scores Grid Search - Random Forest')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # print("-----------------------------------------------------------------------------------")

    # ========== RandomForest normale ========== #
    # Addestramento finale del modello sul training set completo
    forest_model.fit(prices_x_train, prices_y_train)
    
    # Valutazione del modello sul test set
    y_pred = forest_model.predict(prices_x_test)

    # Check dei punteggi
    print(f"MAE RandomForest normale sul test set: {mean_absolute_error(prices_y_test, y_pred):.2f}")
    print(f"MSE RandomForest normale sul test set: {mean_squared_error(prices_y_test, y_pred):.2f}")
    print(f"R2 Score RandomForest normale sul test set: {r2_score(prices_y_test, y_pred):.2f}")
    print("-----------------------------------------------------------------------------------")

    """
    Stampe dei grafici omessi
    plot(y_pred, prices_x_test, prices_y_test, n_feature=0)#sqft_living
    plot(y_pred, prices_x_test, prices_y_test, n_feature=1)#sqft_lot
    plot(y_pred, prices_x_test, prices_y_test, n_feature=2)#floors
    plot(y_pred, prices_x_test, prices_y_test, n_feature=3)#waterfront
    plot(y_pred, prices_x_test, prices_y_test, n_feature=4)#view
    plot(y_pred, prices_x_test, prices_y_test, n_feature=5)#condition
    plot(y_pred, prices_x_test, prices_y_test, n_feature=6)#sqft_above
    plot(y_pred, prices_x_test, prices_y_test, n_feature=7)#sqft_basement
    plot(y_pred, prices_x_test, prices_y_test, n_feature=8)#yr_built
    plot(y_pred, prices_x_test, prices_y_test, n_feature=9)#yr_renovated
    plot(y_pred, prices_x_test, prices_y_test, n_feature=10)#street
    plot(y_pred, prices_x_test, prices_y_test, n_feature=11)#city
    plot(y_pred, prices_x_test, prices_y_test, n_feature=12)#country
    plot(y_pred, prices_x_test, prices_y_test, n_feature=13)#rooms
    """
    # Prediction of the model
    return forest_model

def printConfusionMatrix(targets, predictions, classifierConf, label):
    class_names = np.array(["0","1","2","3","4","5","6"])

    print(F"Classification report for {label}:")
    print(classification_report(targets, predictions, labels=[0,1,2,3,4,5,6], target_names=class_names, digits=3, zero_division="warn"))
    print("============================================")

    cm = confusion_matrix(targets, predictions, labels=classifierConf.classes_)

    return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifierConf.classes_)

def get_Citta():    
    return model.Citta

def get_Regione():
    return model.Regione
    
def get_Via():
    return model.Via

def get_Via_withCity(Citta):
    lista = []
    with open('house.csv') as file_csv:
        reader = csv.DictReader(file_csv)
        for row in reader:
            if row['city'] == Citta:
                lista.append(row['street'])
    lista = sorted(list(dict.fromkeys(lista)))
    return lista

def get_Anno_c():
    return model.Anno_c

def get_Anno_r():
    return model.Anno_r

def get_Living():    
    return model.Living

def get_Lot():    
    return model.Lot

def get_Basement():    
    return model.Basement

def get_WaterFront():    
    return model.WaterFront

def get_Stanze():    
    return model.Stanze

def get_Piani():    
    return model.Piani

def get_Vista():    
    return model.Vista

def get_Condizione():    
    return model.Condizione

def get_Above():    
    return model.Above

def get_Modello():
    return model.Modello