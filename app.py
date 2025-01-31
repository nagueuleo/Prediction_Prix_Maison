#App made with Streamlit

#Loading Modules Needed
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import pickle
import requests
from xgboost import XGBRegressor

#URL of the API made with Flask
API = "http://192.168.0.24:5000"

MODEL_PATH = f'./model/house_price_model.pkl'
SCALER_PATH = f'./model/scaler.pkl'
IMG_SIDEBAR_PATH = "./assets/maison.jpg"

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

#Function to load the Iris Dataset
def get_clean_data():
  data = pd.read_csv("./dataset/regression_multiple.csv")
  data.columns = data.columns.str.strip()
  X = pd.get_dummies(data)

  return X

#Sidebar of the Streamlit App
def add_sidebar():
  st.sidebar.header("Prédiction des Prix des Maisons `App 🏠`")
  image = np.array(Image.open(IMG_SIDEBAR_PATH))
  st.sidebar.image(image)
  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.write("Cette application d'intelligence artificielle peut prédire le prix d'une maison en fonction de ses paramètres correspondants.")

  st.sidebar.subheader('Sélectionnez les paramètres de la maison ✅:')
  
  data = get_clean_data()
  
  slider_labels = [
        ("Superficie ", "Superficie"),
        ("Nombre de chambres", "Nombre de chambres"),
        ("Proximité du centre", "Proximité du centre")  
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

  st.sidebar.markdown('''
  🧑🏻‍💻 Développer par [Lionel NAGUEU](https://github.com/nagueuleo/House_Prediction).
  ''')

  return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['Prix'], axis=1)
  X = pd.get_dummies(X)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

#Radar Chart Function
def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  
  categories = ['Superficie', 'Nombre de chambres', 'Proximité du centre']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['Superficie'], input_data['Nombre de chambres'], input_data['Proximité du centre'],
          
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

#Receiving Prediction Results from the API
def add_predictions(input_data) :
    input_array = np.array(list(input_data.values())).reshape(1, -1).tolist()

    input_array_scaled = scaler.transform(input_array)
    result = model.predict(input_array_scaled)

    pred_result = round(result[0], 2)

    #Run first the api.py file and the paste the URL in the API Variable if you want to deploy the Model with Flask and uncomment the next lines

    data = {'array': input_array}

    resp = requests.post(API, json=data)
    
    pred_result = resp.json()["Results"]["price_result"]
    
    pred_result = round(pred_result, 2)
    pred_result = f"{pred_result}€"

    st.markdown("### Prédiction des prix des maisons 💸")
    st.write("<span class='diagnosis-label diagnosis price'>Model de Machine learning ✅:</span>",  unsafe_allow_html=True)
    
    _, col, _ = st.columns([0.2, 1, 0.2])
    
    with col:
        st.metric("Prix de la maison:", f"{pred_result}", "Euro (€)")

def main() :  
    st.set_page_config(
        page_title="Prédicteur du prix de maisons",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
  
    input_data = add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )    

    with st.container() :
        st.title("Prédiction de prix de maisons")
        st.write("Cette application prédit à l'aide d'un modèle d'apprentissage automatique XGBRegressor le prix en Euro (€) d'une maison. Vous pouvez également mettre à jour les mesures manuellement à l'aide des curseurs de la barre latérale")
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('### Radar des paramètres 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
        st.markdown('### Probabilité du modèle de régression 📉')
        st.image("./assets/probability_plot_model.png")

        st.markdown("---", unsafe_allow_html=True)
        st.write("`Cette intelligence artificielle peut aider à déterminer le prix d'une maison, mais ne doit pas être utilisée comme substitut à un diagnostic et à une prédiction finale.`")


    with col2:
        st.markdown('### Évaluation du modèle 📈')
        st.image("./assets/model_evaluation.png")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        add_predictions(input_data)

        st.markdown("---", unsafe_allow_html=True)
        st.markdown('### Histogramme du modèle 📊')
        st.image("./assets/model_displot.png")
        

if __name__ == "__main__" :
    main()

    print("Execution de l'application!")
