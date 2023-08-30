# -*- coding: utf-8 -*-
import tkinter as tk
import warnings
import numpy as np
import Model as m
from tkinter import ttk
warnings.filterwarnings('ignore')

#Funzione per il pulsante
def predizione_prezzo():

    Label_Prediction.configure(text="")
    
    country = Entry_Country.get()
    index = m.model.dataframe_UI.index[m.model.dataframe_UI['country']==country].tolist()
    country = m.model.dataframe.iloc[index[0],11] # 11 = Country

    street = Entry_Street.get()
    index = m.model.dataframe_UI.index[m.model.dataframe_UI['street']==street].tolist()
    street = m.model.dataframe.iloc[index[0],10] # 10 = Street

    city = Entry_City.get()
    indexCitta = m.model.dataframe.columns.get_loc(F"city_{city}")

    sqft_living = float((Entry_Living.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_living']==sqft_living].tolist()
    #sqft_living = m.model.dataframe.iloc[index[0],0] # 0 = sqft_living

    sqft_lot = float((Entry_Lot.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_lot']==sqft_lot].tolist()
    #sqft_lot = m.model.dataframe.iloc[index[0],1] # 1 = sqft_lot

    sqft_basement = float((Entry_Basement.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_basement']==sqft_basement].tolist()
    #sqft_basement = m.model.dataframe.iloc[index[0],7] # 7 = sqft_basement

    sqft_above = float((Entry_Above.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_above']==sqft_above].tolist()
    #sqft_above = m.model.dataframe.iloc[index[0],6] # 6 = sqft_above

    rooms = float(Entry_Room.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['rooms']==rooms].tolist()
    #rooms = m.model.dataframe.iloc[index[0],12] # 12 = rooms

    floors = float(Entry_Floor.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['floors']==floors].tolist()
    #floors = m.model.dataframe.iloc[index[0],2] # 2 = floors

    waterfront = int(Entry_WF.get())  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['waterfront']==waterfront].tolist()
    #waterfront = m.model.dataframe.iloc[index[0],3] # 3 = waterfront

    view = round(float((Entry_View.get())))  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['view']==view].tolist()
    #view = m.model.dataframe.iloc[index[0],4] # 4 = view

    condition = round(float((Entry_Cond.get())))  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['condition']==condition].tolist()
    #condition = m.model.dataframe.iloc[index[0],5] # 5 = condition

    yr_built = int(Entry_YearC.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['yr_built']==yr_built].tolist()
    #yr_built = m.model.dataframe.iloc[index[0],8] # 8 = yr_built

    yr_renovated = int(Entry_YearR.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['yr_renovated']==yr_renovated].tolist()
    #yr_renovated = m.model.dataframe.iloc[index[0],9] # 9 = yr_renovated
    
    # Prediction
    sample = np.zeros((1,57))
    sample[0,:13] = np.array([sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,country,rooms]).reshape(1, -1)
    sample[0,indexCitta] = 1

    sample = scaler.transform(sample.reshape(1, -1))

    modelScelto = str(ComboBox_Model.get())
    if(modelScelto == 'Random Forest'):
        forest_modelPredict = forest_model.predict(sample)
        Label_Prediction.configure(text=("Il prezzo predetto è: %.2f" %forest_modelPredict))
    elif (modelScelto == 'SGD'):

        predicted_probabilities = SGD_model.predict_proba(sample).squeeze()

        # Retrieve the index and probability
        index = np.argmax(predicted_probabilities) # Indice della probabilità maggiore            
        probability = predicted_probabilities[index]  # Mi prendo la percentuale di probabilità in base all'indice qui sopra

        if(index == len(predicted_probabilities)-1):
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia da {(price_ranges[int(index)])[0]} in su.")
        else:
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia {price_ranges[int(index)]}.")
        Label_Prediction.configure(text=text)
    else:
        raise NotImplementedError(F"Scegli un modello valido. Hai scelto {modelScelto}.")

def update_streets(event):
    streets = m.get_Via_withCity(Entry_City.get())
    Entry_Street.config(values=streets)
    Entry_Street.current(0)
    return

price_ranges = [
    (0, 80000),
    (80000, 150000),
    (150000, 200000),
    (200000, 650000),
    (650000, 1000000),
    (1000000, 3000000),
    (3000000, float('inf'))
]

prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler  = m.crea_basedati(modelUsed="RandomForest")
prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD, scaler_SGD  = m.crea_basedati(modelUsed="SGD")

forest_model = m.modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test)
SGD_model = m.modello2(prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD)

window = tk.Tk()

window.geometry("700x700")
window.title("House Prediction in USA")
window.resizable(False, False)

Label_Titolo = tk.Label(window, text="Seleziona i valori", font=("Helvetica", 20))
Label_Titolo.grid(row=0,column=0, columnspan=3, padx=10)

#Creo l'etichetta e la combobox per il valore "Country"
Label_Country = tk.Label(window, text = "Nazione: ", font=("Helvetica", 16))
Label_Country.grid(row=1, column=0, padx=1, sticky="W")
Entry_Country = ttk.Combobox(window, values = m.get_Regione(), state = 'readonly')
Entry_Country.grid(row=1, column=1, padx=1, sticky="W")
Entry_Country.current(0)

#Creo l'etichetta e la combobox per il valore "City"
Label_City = tk.Label(window, text = "Citta': ", font=("Helvetica", 16))
Label_City.grid(row=2, column=0, padx=1, sticky="W")
Entry_City = ttk.Combobox(window, values = m.get_Citta(), state = 'readonly')
Entry_City.grid(row=2, column=1, padx=1, sticky="W")
Entry_City.current(0)

#Creo l'etichetta e la combobox per il valore "Street"
Label_Street = tk.Label(window, text = "Via: ", font=("Helvetica", 16))
Label_Street.grid(row=3, column=0, padx=1, sticky="W")
Entry_Street = ttk.Combobox(window, values = m.get_Via(), state = 'readonly')
Entry_Street.grid(row=3, column=1, padx=1, sticky="W")
Entry_Street.current(0)
Entry_City.bind("<<ComboboxSelected>>", update_streets) # Quando selezioniamo la città, aggiorniamo le strade

#Creo l'etichetta e la combobox per il valore "Sqft_Living"
Label_Living = tk.Label(window, text = "Metri quadri Vivibili: ", font=("Helvetica", 16))
Label_Living.grid(row=4, column=0, padx=1, sticky="W")
Entry_Living = ttk.Combobox(window, values = m.get_Living(), state = 'normal')
Entry_Living.grid(row=4, column=1, padx=1, sticky="W")

#Creo l'etichetta e la combobox per il valore "Sqft_Lot"
Label_Lot = tk.Label(window, text = "Metri quadri Lotto: ", font=("Helvetica", 16))
Label_Lot.grid(row=5, column=0, padx=1, sticky="W")
Entry_Lot = ttk.Combobox(window, values = m.get_Lot(), state = 'normal')
Entry_Lot.grid(row=5, column=1, padx=1, sticky="W")

#Creo l'etichetta e la combobox per il valore "Sqft_Basement"
Label_Basement = tk.Label(window, text = "Metri quadri Seminterrato: ", font=("Helvetica", 16))
Label_Basement.grid(row=6, column=0, padx=1, sticky="W")
Entry_Basement = ttk.Combobox(window, values = m.get_Basement(), state = 'normal')
Entry_Basement.grid(row=6, column=1, padx=1, sticky="W")

#Creo l'etichetta e la combobox per il valore "Sqft_Above"
Label_Above = tk.Label(window, text = "Metri quadri Calpestabili: ", font=("Helvetica", 16))
Label_Above.grid(row=7, column=0, padx=1, sticky="W")
Entry_Above = ttk.Combobox(window, values = m.get_Above(), state = 'normal')
Entry_Above.grid(row=7, column=1, padx=1, sticky="W")

Entry_Living.current(0)
Entry_Lot.current(0)
Entry_Basement.current(0)
Entry_Above.current(0)

#Creo l'etichetta e la combobox per il valore "YearC"
Label_YearC = tk.Label(window, text = "Anno di costruzione: ", font=("Helvetica", 16))
Label_YearC.grid(row=8, column=0, padx=1, sticky="W")
Entry_YearC = ttk.Combobox(window, values = m.get_Anno_c(), state = 'readonly')
Entry_YearC.grid(row=8, column=1, padx=1, sticky="W")
Entry_YearC.current(0)

#Creo l'etichetta e la combobox per il valore "YearR"
Label_YearR = tk.Label(window, text = "Anno di restauro: ", font=("Helvetica", 16))
Label_YearR.grid(row=9, column=0, padx=1, sticky="W")
Entry_YearR = ttk.Combobox(window, values = m.get_Anno_r(), state = 'readonly')
Entry_YearR.grid(row=9, column=1, padx=1, sticky="W")
Entry_YearR.current(0)

#Creo l'etichetta e lo spinbox per il valore "Floor"
Label_Floor = tk.Label(window, text = "Piani: ", font=("Helvetica", 16))
Label_Floor.grid(row=10, column=0, padx=1, sticky="W")
Entry_Floor = ttk.Spinbox(window,from_ = 0, to = 4, wrap=True, increment=0.5, format='%1.1f', state = 'readonly')
Entry_Floor.grid(row=10, column=1, padx=1, sticky="W")
Entry_Floor.set(1.0)

#Creo l'etichetta e lo spinbox per il valore "WaterFront"
Label_WF = tk.Label(window, text = "Affaccio sul mare: ", font=("Helvetica", 16))
Label_WF.grid(row=11, column=0, padx=1, sticky="W")
Entry_WF = ttk.Spinbox(window,from_ = 0, to = 1, wrap=True, increment=1, format='%1d', state = 'readonly')
Entry_WF.grid(row=11, column=1, padx=1, sticky="W")
Entry_WF.set(0)

#Creo l'etichetta e lo spinbox per il valore "Rooms"
Label_Room = tk.Label(window, text = "Stanze: ", font=("Helvetica", 16))
Label_Room.grid(row=12, column=0, padx=1, sticky="W")
Entry_Room = ttk.Spinbox(window,from_ = 1, to = 10, wrap=True, increment=0.25, format='%1.2f', state = 'readonly')
Entry_Room.grid(row=12, column=1, padx=1, sticky="W")
Entry_Room.set(2.0)

#Creo l'etichetta e lo spinbox per il valore "View"
Label_View = tk.Label(window, text = "Vista: ", font=("Helvetica", 16))
Label_View.grid(row=13, column=0, padx=1, sticky="W")
Entry_View = ttk.Spinbox(window,from_ = 0, to = 5, wrap=True, increment=1, format='%1.1f', state = 'readonly')
Entry_View.grid(row=13, column=1, padx=1, sticky="W")
Entry_View.set(0)

#Creo l'etichetta e lo spinbox per il valore "Condition"
Label_Cond = tk.Label(window, text = "Condizione: ", font=("Helvetica", 16))
Label_Cond.grid(row=14, column=0, padx=1, sticky="W")
Entry_Cond = ttk.Spinbox(window,from_ = 1, to = 5, wrap=True, increment=1, format='%1.1f', state = 'readonly')
Entry_Cond.grid(row=14, column=1, padx=1, sticky="W")
Entry_Cond.set(0)

# ComboBox per il modello
Label_Model = tk.Label(window, text = "Scegli il modello: ", font=("Helvetica", 16))
Label_Model.grid(row=17, column=0, padx=1, sticky="W")
ComboBox_Model = ttk.Combobox(window, values = m.get_Modello(), state = 'readonly')
ComboBox_Model.grid(row=17, column=1, padx=1, sticky="W")
ComboBox_Model.current(0)

#Creo il pulsante per il passaggio dei valori dell'utente
getValue_button = tk.Button(text="Avvia Predizione", command=predizione_prezzo)
getValue_button.grid(row=18, column=0, columnspan=2, padx=10, pady=10)

# Label per la predizione
Label_Prediction = tk.Label(window, text="", fg="red",font=("Helvetica", 16))
Label_Prediction.grid(row=20, column=0,columnspan=2, padx=10) 

if __name__ == "__main__":
    window.mainloop()
