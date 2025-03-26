import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# Configurar la pagina
st.set_page_config(page_title='Predicción de riesgo de seguros', page_icon='🚗', layout='centered', initial_sidebar_state='auto')
st.image("Logo_seguro_carro.jpg", width=800)

#punto de entrada
def main():
    # Cargar el modelo
    filename = "modelo-clas-tree-RL-Knn-SVM-RF.pkl"
    #modelTree, labelencoder, variables = pickle.load(open(filename, 'rb')) #rb --> Modo Lectura  Arbol de desición
    modelTree, model_RL, model_knn, model_SVM,model_RF, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb')) #rb --> Modo Lectura para cargar los objetos con: modelTree, model_RL, labelencoder, variables, min_max_scaler
    
    #titulo principal
   
    #st.title('Predicción de riesgo de seguros') 
    #Titulo sidebar
    st.sidebar.title('Ingresar datos del cliente')
    #variables

    # Entradas del usuario en el sidebar
    def user_input_features():
        #edad como  valor entero entre 0 y 50 años
        edad = st.sidebar.slider('Edad', min_value=18, max_value=50, value= 25, step=1) # Step =1 para que se mueva de 1 en 1

        # Entradas variable cartype
        options = ["combi", "minivan","sport","family"]
        cartype = st.sidebar.selectbox('Tipo de vehículo', options)
        
        #Crear diccionario data con los valores de entrada
        data = {'age': edad,
                'cartype': cartype}
        #Crear un DataFrame a partir de los datos
        features = pd.DataFrame(data, index=[0])
        #st.caption('Datos del cliente:')
        #st.write(features)
       
        #Preparar los datos de entrada
        data_preparada = features.copy()
        #st.write(data_preparada)

        #Crear  la variables dummies de la variable cartype
        data_preparada = pd.get_dummies(data_preparada,columns=['cartype'], drop_first=False)
        #st.caption('Datos del cliente con dummies:')
        #st.write(data_preparada)
        # Realizar reindexación para añadir columnas faltantes  en el caso de que no se seleccionen todas las opciones
        data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
        #st.caption('Datos del cliente con reindex:')
        #st.write(data_preparada)

        return data_preparada
    #Llamada de la función user_input_features()
    df = user_input_features()
    
    #Selector de modelos
    options = ["DT", "RL", "Knn", "SVM", "RF"]
    model = st.sidebar.selectbox('Seleccionar modelo', options)
    #st.caption('Modelo seleccionado:')
    #st.write(model)

    #Botón de predicción
    if st.button('Realizar Predicción'):
        if model == "DT":
            y_fut = modelTree.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo del cliente es {}'.format(resultado[0]))
        elif model == "RF":
            y_fut = model_RF.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo del cliente es {}'.format(resultado[0]))
        elif model == "RL":
            df["age"] = min_max_scaler.transform(df[["age"]])
            #write = st.write(df)
            y_fut = model_RL.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo del cliente es {}'.format(resultado[0]))
        elif model == "Knn":
            df["age"] = min_max_scaler.transform(df[["age"]])  
            #write = st.write(df)
            y_fut = model_knn.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo del cliente es {}'.format(resultado[0]))
        elif model == "SVM":
            df["age"] = min_max_scaler.transform(df[["age"]])   
            y_fut = model_SVM.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo del cliente es {}'.format(resultado[0]))
    

    
if __name__ == '__main__':
    main()
