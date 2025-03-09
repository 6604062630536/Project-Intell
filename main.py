import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , r2_score
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
import cv2 # type: ignore
from tensorflow import keras
from tensorflow.keras.models import Sequential , load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Dropout , BatchNormalization , Input # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers , models # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping ,LearningRateScheduler # type: ignore
import requests 
import plotly.graph_objects as go
from tensorflow.keras.layers import LeakyReLU # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore

# ฟังก์ชันหลักที่เลือกหน้าจากเมนู
def home():

    st.title("Predict total game sales using Machine Learning")
    st.subheader("Purpose")
    purpose = '''There are many different types of games available worldwide, and the variety of games attracts a diverse group of customers. 
        While some games achieve high sales, others perform poorly. 
        Several factors contribute to this, such as the genre of the game, the player's country, the game's price, and the platform. 
        This model aims to :green[Predict the Global Sales of a game based on characteristics such as release year, genre, platform, publisher, etc.], providing valuable insights for game developers.'''
    st.write(purpose)

    st.subheader("Market Analysis & Trends:")
    market = '''
        • Identify best-selling genres and platforms over the years.

        • Analyze the impact of publishers on sales performance.

        • Track sales trends across regions (NA, EU, JP, Other).
        '''
    st.markdown(market)

    st.subheader("Predictive Analytics & Forecasting:")
    preanal = '''
        • Build models to predict future game sales based on historical data.

        • Identify factors influencing high sales: platform, genre, publisher.'''
    st.markdown(preanal)

    st.subheader("Business Strategy & Game Development:")
    strategy = '''
        • Help game developers understand what type of games sell best.

        • Guide marketing strategies by targeting specific regions with high demand.'''
    st.markdown(strategy)

    st.subheader("Comparative Analysis:")
    comparative = '''
        • Compare sales performance of different gaming platforms (PlayStation, Xbox, Wii).

        • Evaluate how different game publishers perform in various markets.'''
    st.markdown(comparative)

    st.subheader("Problem Type")
    problem_type = '''
        • **Regression Problem**: Predict the total sales (global or regional) of a game based on its attributes, using numerical values for sales predictions.'''
    st.markdown(problem_type)

    st.subheader("Data Used")
    data_used = '''
        The dataset contains various features related to video game sales and characteristics. The features include:

        - **Rank**: Ranking of overall sales.
        - **Name**: The game's name.
        - **Platform**: Platform of the game's release (e.g., PC, PS4).
        - **Year**: Year of the game's release.
        - **Genre**: Genre of the game (e.g., Action, Adventure, RPG).
        - **Publisher**: Publisher of the game.
        - **NA_Sales**: Sales in North America (in millions).
        - **EU_Sales**: Sales in Europe (in millions).
        - **JP_Sales**: Sales in Japan (in millions).
        - **Other_Sales**: Sales in the rest of the world (in millions).
        - **Global_Sales**: Total worldwide sales (in millions).
        '''
    st.markdown(data_used)

    st.subheader("Data Exploration") 
    st.write("The dataset come form :blue[https://www.kaggle.com/gregorut/videogamesales].")
    st.write("The dataset contains :green[16,598] rows and :green[11 columns.]")
    st.subheader("Missing Values")
    st.write("Year has :red[271] missing values. Publisher has :red[58] missing values.")
    st.subheader("Data Cleaning")
    st.write("The dataset has been cleaned and preprocessed to handle missing values and ensure data quality by"
    " :green[Interpolating missing values in the 'Year' column.]"
    " :red[Dropping rows with missing values in the 'Publisher' column.]")
    st.subheader("Using Models")
    st.write("Two models are used to predict the total sales of video games:")
    st.write("1. :green[Random Forest Regressor]: A machine learning model that uses an ensemble of decision trees to predict sales.")
    st.write("2. :green[Linear Regression]: A statistical model that predicts sales based on linear relationships between features.")
    st.write("The models are trained on the dataset and evaluated based on their performance metrics.")
    st.write("The model with the lowest Mean Absolute Error (MAE) and highest R² Score is considered the best model for predicting sales.")
    st.write("The model's performance is visualized using scatter plots to compare actual sales with predicted sales.")

    st.subheader("Random Forest Model Building")
    model_building = ''' # โหลดข้อมูล
    df = pd.read_csv("vgsales.csv")

    # ล้างข้อมูล
    df["Year"] = df["Year"].interpolate(method="linear")
    df.dropna(subset=["Publisher"], inplace=True)
    df["Year"] = df["Year"].astype(int)

    # กรองข้อมูลเมื่อกดปุ่ม Apply Filter
    if apply_filter:
        # กรองข้อมูลตามที่ผู้ใช้เลือก
        filtered_df = df[df["Year"] == selected_year]
        if selected_genre != "All":
            filtered_df = filtered_df[filtered_df["Genre"] == selected_genre]
        if selected_platform != "All":
            filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]

        # ถ้าไม่มีข้อมูลที่กรอง
        if filtered_df.empty:
            st.warning("❌ No data available for the selected filters.")
        else:
            # แสดงผลลัพธ์ที่กรองแล้ว
            st.write(f"📌 Showing results for **Genre: {selected_genre}**, **Platform: {selected_platform}**, **Year: {selected_year}**")
            st.dataframe(filtered_df)

            # 🔹 แปลงค่าหมวดหมู่เป็นตัวเลข
            filtered_df["Genre"] = filtered_df["Genre"].map({
                "Action": 1, "Adventure": 2, "Fighting": 3, "Misc": 4,
                "Platform": 5, "Puzzle": 6, "Racing": 7, "Role-Playing": 8,
                "Shooter": 9, "Simulation": 10, "Sports": 11, "Strategy": 12
            })

            filtered_df["Platform"] = filtered_df["Platform"].map({
                "2600": 1, "3DO": 2, "3DS": 3, "DC": 4, "DS": 5, "GB": 6,
                "GBA": 7, "GC": 8, "GEN": 9, "GG": 10, "N64": 11, "NES": 12,
                "NG": 13, "PC": 14, "PCFX": 15, "PS": 16, "PS2": 17, "PS3": 18,
                "PS4": 19, "PSP": 20, "PSV": 21, "SAT": 22, "SCD": 23, "SNES": 24,
                "TG16": 25, "WS": 26, "Wii": 27, "WiiU": 28, "X360": 29, "XB": 30, "XOne": 31
            })

            filtered_df["Publisher"] = filtered_df["Publisher"].astype("category").cat.codes

            # 🔹 แบ่งข้อมูลเป็น Features และ Target
            X = filtered_df.drop(columns=["Name", "Global_Sales", "Rank"]) 
            y = filtered_df["Global_Sales"]

            # 🔹 แบ่งข้อมูลเป็นชุด Train และ Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 🔹 โมเดลแรก - Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # 🔹 ทำนายค่า
            rf_y_pred = rf_model.predict(X_test)

            # 🔹 คำนวณค่าความแม่นยำ (Accuracy) และ Loss
            rf_mae = mean_absolute_error(y_test, rf_y_pred)
            rf_r2 = r2_score(y_test, rf_y_pred)
            rf_loss = rf_mae / y_test.mean()  # Loss = MAE เทียบกับค่าเฉลี่ยยอดขายจริง

            # 🔹 แสดงผลลัพธ์
            st.subheader("📊 Random Forest Model Performance")
            st.write(f"✅ **Mean Absolute Error (MAE):** {rf_mae:.2f}")
            st.write(f"✅ **Model Accuracy (R² Score):** {rf_r2:.2f}")
            st.write(f"✅ **Loss (MAE / Avg Sales):** {rf_loss:.4f}")

            # 🔹 สร้าง DataFrame สำหรับผลการทำนายของ Random Forest
            rf_result_df = pd.DataFrame({"Actual": y_test, "Predicted": rf_y_pred, "Game": filtered_df.loc[y_test.index, "Name"]})

            # 🔹 สร้างแผนภูมิการพยากรณ์ยอดขายสำหรับ Random Forest
            rf_fig = px.scatter(rf_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Global Sales - Random Forest",
                                labels={"Actual": "Actual Sales (millions)", "Predicted": "Predicted Sales (millions)", "Game": "Game Name"},
                                hover_data=["Game"],
                                template="plotly_dark")
            st.plotly_chart(rf_fig)'''
    st.code(model_building, language="python")

    st.subheader("Linear Regression Model Building")
    model_building = ''' # โมเดลที่สอง - Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)

            # 🔹 ทำนายค่า
            lr_y_pred = lr_model.predict(X_test)

            # 🔹 คำนวณค่าความแม่นยำ (Accuracy) และ Loss
            lr_mae = mean_absolute_error(y_test, lr_y_pred)
            lr_r2 = r2_score(y_test, lr_y_pred)
            lr_loss = lr_mae / y_test.mean()  # Loss = MAE เทียบกับค่าเฉลี่ยยอดขายจริง

            # 🔹 แสดงผลลัพธ์
            st.subheader("📊 Linear Regression Model Performance")
            st.write(f"✅ **Mean Absolute Error (MAE):** {lr_mae:.2f}")
            st.write(f"✅ **Model Accuracy (R² Score):** {lr_r2:.2f}")
            st.write(f"✅ **Loss (MAE / Avg Sales):** {lr_loss:.4f}")

            # 🔹 สร้าง DataFrame สำหรับผลการทำนายของ Linear Regression
            lr_result_df = pd.DataFrame({"Actual": y_test, "Predicted": lr_y_pred, "Game": filtered_df.loc[y_test.index, "Name"]})

            # 🔹 สร้างแผนภูมิการพยากรณ์ยอดขายสำหรับ Linear Regression
            lr_fig = px.scatter(lr_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Global Sales - Linear Regression",
                                labels={"Actual": "Actual Sales (millions)", "Predicted": "Predicted Sales (millions)", "Game": "Game Name"},
                                hover_data=["Game"],
                                template="plotly_dark")
            st.plotly_chart(lr_fig)'''
    st.code(model_building, language="python")
    # โหลดข้อมูล
    df = pd.read_csv("vgsales.csv")
    df["Year"] = df["Year"].interpolate(method="linear")
    df.dropna(subset=["Publisher"], inplace=True)

    NA = px.scatter(df, 
                    x="NA_Sales",  # 
                    y="Genre",  # แสดงภูมิภาค
                    title="Video Game Sales by Genre and Region",
                    color="Platform",  # แสดงข้อมูลแยกตาม Platform หรือเลือกตามภูมิภาค
                    size="Global_Sales",  # ขนาดของจุดตามยอดขาย
                    hover_name="Name", 
                    hover_data=["Publisher", "Year"],  # ข้อมูลเพิ่มเติมเมื่อชี้เมาส์
                    labels={"Global_Sales": "Sales (in millions)", "Genre": "Game Genre"},
                    template="plotly_dark")  # ใช้เทมเพลต Dark

    st.plotly_chart(NA)

    JP = px.scatter(df, 
                    x="JP_Sales",  # 
                    y="Genre",  # แสดงภูมิภาค
                    title="Video Game Sales by Genre and Region",
                    color="Platform",  # แสดงข้อมูลแยกตาม Platform หรือเลือกตามภูมิภาค
                    size="Global_Sales",  # ขนาดของจุดตามยอดขาย
                    hover_name="Name", 
                    hover_data=["Publisher", "Year"],  # ข้อมูลเพิ่มเติมเมื่อชี้เมาส์
                    labels={"Global_Sales": "Sales (in millions)", "Genre": "Game Genre"},
                    template="plotly_dark")  # ใช้เทมเพลต Dark

    st.plotly_chart(JP)

    EU = px.scatter(df, 
                    x="EU_Sales",  # 
                    y="Genre",  # แสดงภูมิภาค
                    title="Video Game Sales by Genre and Region",
                    color="Platform",  # แสดงข้อมูลแยกตาม Platform หรือเลือกตามภูมิภาค
                    size="Global_Sales",  # ขนาดของจุดตามยอดขาย
                    hover_name="Name", 
                    hover_data=["Publisher", "Year"],  # ข้อมูลเพิ่มเติมเมื่อชี้เมาส์
                    labels={"Global_Sales": "Sales (in millions)", "Genre": "Game Genre"},
                    template="plotly_dark")  # ใช้เทมเพลต Dark

    st.plotly_chart(EU)

    other = px.scatter(df, 
                    x="Other_Sales",  # 
                    y="Genre",  # แสดงภูมิภาค
                    title="Video Game Sales by Genre and Region",
                    color="Platform",  # แสดงข้อมูลแยกตาม Platform หรือเลือกตามภูมิภาค
                    size="Global_Sales",  # ขนาดของจุดตามยอดขาย
                    hover_name="Name", 
                    hover_data=["Publisher", "Year"],  # ข้อมูลเพิ่มเติมเมื่อชี้เมาส์
                    labels={"Global_Sales": "Sales (in millions)", "Genre": "Game Genre"},
                    template="plotly_dark")  # ใช้เทมเพลต Dark

    st.plotly_chart(other)

def vgsales():
    st.title("🎮 Predict total game sales")

    # โหลดข้อมูล
    df = pd.read_csv("vgsales.csv")

    # ล้างข้อมูล
    df["Year"] = df["Year"].interpolate(method="linear")
    df.dropna(subset=["Publisher"], inplace=True)
    df["Year"] = df["Year"].astype(int)

    # **สร้าง Widget สำหรับเลือก Genre และ Platform**
    genre_options = df["Genre"].unique()
    platform_options = df["Platform"].unique()
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())

    # Dropdown สำหรับเลือก Genre & Platform
    selected_genre = st.selectbox("🎮 Select Genre", ["All"] + list(genre_options))
    selected_platform = st.selectbox("🕹 Select Platform", ["All"] + list(platform_options))

    # Slider สำหรับเลือก Year
    selected_year = st.slider("📅 Select Year", min_value=year_min, max_value=year_max, value=year_max)
    st.write("if No data available for the selected filters, please try different filters.")
    # ปุ่ม Apply Filter
    apply_filter = st.button("Apply Filter")

    # กรองข้อมูลเมื่อกดปุ่ม Apply Filter
    if apply_filter:
        # กรองข้อมูลตามที่ผู้ใช้เลือก
        filtered_df = df[df["Year"] == selected_year]
        if selected_genre != "All":
            filtered_df = filtered_df[filtered_df["Genre"] == selected_genre]
        if selected_platform != "All":
            filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]

        # ถ้าไม่มีข้อมูลที่กรอง
        if filtered_df.empty:
            st.warning("❌ No data available for the selected filters.")
        else:
            # แสดงผลลัพธ์ที่กรองแล้ว
            st.write(f"📌 Showing results for **Genre: {selected_genre}**, **Platform: {selected_platform}**, **Year: {selected_year}**")
            st.dataframe(filtered_df)

            # 🔹 แปลงค่าหมวดหมู่เป็นตัวเลข
            filtered_df["Genre"] = filtered_df["Genre"].map({
                "Action": 1, "Adventure": 2, "Fighting": 3, "Misc": 4,
                "Platform": 5, "Puzzle": 6, "Racing": 7, "Role-Playing": 8,
                "Shooter": 9, "Simulation": 10, "Sports": 11, "Strategy": 12
            })

            filtered_df["Platform"] = filtered_df["Platform"].map({
                "2600": 1, "3DO": 2, "3DS": 3, "DC": 4, "DS": 5, "GB": 6,
                "GBA": 7, "GC": 8, "GEN": 9, "GG": 10, "N64": 11, "NES": 12,
                "NG": 13, "PC": 14, "PCFX": 15, "PS": 16, "PS2": 17, "PS3": 18,
                "PS4": 19, "PSP": 20, "PSV": 21, "SAT": 22, "SCD": 23, "SNES": 24,
                "TG16": 25, "WS": 26, "Wii": 27, "WiiU": 28, "X360": 29, "XB": 30, "XOne": 31
            })

            filtered_df["Publisher"] = filtered_df["Publisher"].astype("category").cat.codes

            # 🔹 แบ่งข้อมูลเป็น Features และ Target
            X = filtered_df.drop(columns=["Name", "Global_Sales", "Rank"]) 
            y = filtered_df["Global_Sales"]

            # 🔹 แบ่งข้อมูลเป็นชุด Train และ Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 🔹 โมเดลแรก - Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # 🔹 ทำนายค่า
            rf_y_pred = rf_model.predict(X_test)

            # 🔹 คำนวณค่าความแม่นยำ (Accuracy) และ Loss
            rf_mae = mean_absolute_error(y_test, rf_y_pred)
            rf_r2 = r2_score(y_test, rf_y_pred)
            rf_loss = rf_mae / y_test.mean()  # Loss = MAE เทียบกับค่าเฉลี่ยยอดขายจริง

            # 🔹 แสดงผลลัพธ์
            st.subheader("📊 Random Forest Model Performance")
            st.write(f"✅ **Mean Absolute Error (MAE):** {rf_mae:.2f}")
            st.write(f"✅ **Model Accuracy (R² Score):** {rf_r2:.2f}")
            st.write(f"✅ **Loss (MAE / Avg Sales):** {rf_loss:.4f}")

            # 🔹 สร้าง DataFrame สำหรับผลการทำนายของ Random Forest
            rf_result_df = pd.DataFrame({"Actual": y_test, "Predicted": rf_y_pred, "Game": filtered_df.loc[y_test.index, "Name"]})

            # 🔹 สร้างแผนภูมิการพยากรณ์ยอดขายสำหรับ Random Forest
            rf_fig = px.scatter(rf_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Global Sales - Random Forest",
                                labels={"Actual": "Actual Sales (millions)", "Predicted": "Predicted Sales (millions)", "Game": "Game Name"},
                                hover_data=["Game"],
                                template="plotly_dark")
            st.plotly_chart(rf_fig)

            # 🔹 โมเดลที่สอง - Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)

            # 🔹 ทำนายค่า
            lr_y_pred = lr_model.predict(X_test)

            # 🔹 คำนวณค่าความแม่นยำ (Accuracy) และ Loss
            lr_mae = mean_absolute_error(y_test, lr_y_pred)
            lr_r2 = r2_score(y_test, lr_y_pred)
            lr_loss = lr_mae / y_test.mean()  # Loss = MAE เทียบกับค่าเฉลี่ยยอดขายจริง

            # 🔹 แสดงผลลัพธ์
            st.subheader("📊 Linear Regression Model Performance")
            st.write(f"✅ **Mean Absolute Error (MAE):** {lr_mae:.2f}")
            st.write(f"✅ **Model Accuracy (R² Score):** {lr_r2:.2f}")
            st.write(f"✅ **Loss (MAE / Avg Sales):** {lr_loss:.4f}")

            # 🔹 สร้าง DataFrame สำหรับผลการทำนายของ Linear Regression
            lr_result_df = pd.DataFrame({"Actual": y_test, "Predicted": lr_y_pred, "Game": filtered_df.loc[y_test.index, "Name"]})

            # 🔹 สร้างแผนภูมิการพยากรณ์ยอดขายสำหรับ Linear Regression
            lr_fig = px.scatter(lr_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Global Sales - Linear Regression",
                                labels={"Actual": "Actual Sales (millions)", "Predicted": "Predicted Sales (millions)", "Game": "Game Name"},
                                hover_data=["Game"],
                                template="plotly_dark")
            st.plotly_chart(lr_fig)

def pokedex():
    st.title("Predict Pokemon")
    st.subheader("Purpose")
    purpose = '''The goal of this project is to predict the outcome of a Pokémon battle based on various attributes such as HP, Attack, Defense, Speed, and Type Advantage.
This model helps trainers analyze the strengths and weaknesses of Pokémon, allowing them to make better strategic decisions during battles.'''
    st.write(purpose)
    
    st.subheader("Problem Type")
    problem_type = '''
       Binary Classification Problem: Predict which Pokémon is more likely to win in a battle based on their attributes.'''
    st.markdown(problem_type)

    st.subheader("Data Used")
    data_used = '''
        The dataset contains various features related to Pokemon attributes and evolution. The features include:
        - **ID**: Unique identifier for each Pokemon.
        - **Name**: Name of the Pokemon.
        - **Type 1**: Primary type of the Pokemon (e.g., Fire, Water, Grass).
        - **Type 2**: Secondary type of the Pokemon (some Pokemon don't have a secondary type).
        - **Total**: Total number of stats for the Pokemon.
        - **HP**: Health points of the Pokemon.
        - **Attack**: Attack power of the Pokemon.
        - **Defense**: Defense power of the Pokemon.
        - **Speed**: Speed of the Pokemon. '''
    st.markdown(data_used)
    
    st.subheader("Data Exploration")
    st.write("The dataset comes from :blue[https://www.kaggle.com/datasets/abcsds/pokemon/data].")
    st.write("The dataset contains :green[721] rows and :green[9 columns].")
    st.subheader("Missing Values")
    st.write("There may be missing values in columns such as **Type 2**, which may not be applicable for all Pokémon.")
    st.code("""if pd.isna(attack_type):  # ตรวจสอบว่า attack_type เป็น NaN หรือไม่
            continue  # ข้ามการคำนวณถ้าเป็น NaN
        for defense_type in [opponent_type1, opponent_type2]:
            if pd.isna(defense_type):  # ตรวจสอบว่า defense_type เป็น NaN หรือไม่
                continue  # ข้ามการคำนวณถ้าเป็น NaN""")
    st.subheader("Data Cleaning")
    st.write("The dataset has been cleaned and preprocessed to handle missing values and ensure data quality by")
    st.code("""def clean_data(data):
    data = data[['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    data = data[~data['Name'].str.contains('Mega')]
    data['Total'] = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)
    return data""")
    
    st.subheader("Using Models")
    st.write("A Neural Network (Deep Learning) model is used to learn battle patterns."
        'The model consists of Dense (Fully Connected) Layers with Dropout to reduce overfitting.'
        'StandardScaler is applied to normalize the feature values for better model performance.'
        "The model is trained using HP, Attack, Defense, Speed, Sp. Atk, Sp. Def, Total, and Type Advantage as input features.")
    st.code ("""# โหลดข้อมูลจากไฟล์ CSV
@st.cache_data
def load_data():
    return pd.read_csv('pokemon.csv')

@st.cache_data
def load_type_effectiveness():
    return pd.read_csv('type_effectiveness.csv')

# ดึง URL รูปภาพโปเกมอนจาก PokéAPI
def get_pokemon_image_url(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['sprites']['front_default']
    return None

# คำนวณ Type Advantage
def calculate_type_advantage(type1, type2, opponent_type1, opponent_type2, type_effectiveness):
    advantage = 1.0
    for attack_type in [type1, type2]:
        if pd.isna(attack_type):
            continue
        for defense_type in [opponent_type1, opponent_type2]:
            if pd.isna(defense_type):
                continue
            effectiveness = type_effectiveness[
                (type_effectiveness['Attacking Type'] == attack_type) &
                (type_effectiveness['Defending Type'] == defense_type)
            ]['Effectiveness'].values
            if len(effectiveness) > 0:
                advantage *= effectiveness[0]
    return advantage

# ทำความสะอาดข้อมูล
def clean_data(data):
    data = data[['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    data = data[~data['Name'].str.contains('Mega')]
    data['Total'] = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)
    return data

# สร้างโมเดล Neural Network
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ทำนายผลการต่อสู้
def predict_battle(model, pokemon1_stats, pokemon2_stats, type_advantage, scaler):
    features = np.concatenate([pokemon1_stats, pokemon2_stats, [type_advantage]])
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0][0]

# Streamlit App
def predict_pokemon():
    st.title("Pokemon Battle Predictor ⚔️")
    data = clean_data(load_data())
    type_effectiveness = load_type_effectiveness()
    pokemon_list = data['Name'].unique()
    
    pokemon1 = st.selectbox("Select Pokemon 1", pokemon_list)
    pokemon2 = st.selectbox("Select Pokemon 2", pokemon_list)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{pokemon1}**")
        image_url1 = get_pokemon_image_url(pokemon1)
        if image_url1:
            st.image(image_url1, width=150)
    with col2:
        st.write(f"**{pokemon2}**")
        image_url2 = get_pokemon_image_url(pokemon2)
        if image_url2:
            st.image(image_url2, width=150)
    
    if pokemon1 and pokemon2:
        pokemon1_data = data[data['Name'] == pokemon1].iloc[0]
        pokemon2_data = data[data['Name'] == pokemon2].iloc[0]
        type_advantage = calculate_type_advantage(
            pokemon1_data['Type 1'], pokemon1_data['Type 2'],
            pokemon2_data['Type 1'], pokemon2_data['Type 2'],
            type_effectiveness
        )
        
        X = pd.concat([data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']]] * 2, axis=1)
        X['Type Advantage'] = 1.0
        y = np.random.randint(2, size=len(X))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        pokemon1_stats = pokemon1_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']].values
        pokemon2_stats = pokemon2_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']].values
        prediction = predict_battle(model, pokemon1_stats, pokemon2_stats, type_advantage, scaler)
        
        st.subheader("🎮 Battle Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{pokemon1}** Stats:")
            for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
                st.write(f"{stat}: {pokemon1_data[stat]}")
        with col2:
            st.write(f"**{pokemon2}** Stats:")
            for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
                st.write(f"{stat}: {pokemon2_data[stat]}")
        
        st.write(f"**Type Advantage:** {type_advantage}")
        
        if prediction > 0.5:
            st.success(f"{pokemon1} Wins!")
        else:
            st.success(f"{pokemon2} Wins!")
        
        accuracy, loss = model.evaluate(X_test, y_test, verbose=0)
        st.subheader("📊 Model Performance")
        st.write(f"✅ **Accuracy:** {accuracy:.2f}")
        st.write(f"✅ **Loss:** {loss:.4f}")
""")


# โหลดข้อมูลจาก CSV
@st.cache_data
def load_data():
    return pd.read_csv('pokemon.csv')

@st.cache_data
def load_type_effectiveness():
    return pd.read_csv('type_effectiveness.csv')

# ดึง URL รูปภาพโปเกมอนจาก PokéAPI
def get_pokemon_image_url(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['sprites']['front_default']
    return None

# คำนวณ Type Advantage
def calculate_type_advantage(type1, type2, opponent_type1, opponent_type2, type_effectiveness):
    advantage = 1.0
    for attack_type in [type1, type2]:
        if pd.isna(attack_type):
            continue
        for defense_type in [opponent_type1, opponent_type2]:
            if pd.isna(defense_type):
                continue
            effectiveness = type_effectiveness[
                (type_effectiveness['Attacking Type'] == attack_type) & 
                (type_effectiveness['Defending Type'] == defense_type)
            ]['Effectiveness'].values
            if len(effectiveness) > 0:
                if effectiveness[0] == 0:
                    return 0.0
                advantage *= effectiveness[0]
    return advantage

def calculate_battle_advantage(pokemon1_data, pokemon2_data, type_advantage):
    # คำนวณความได้เปรียบจาก Type
    type_adv = type_advantage

    # คำนวณความได้เปรียบจาก Attack/Defense
    # สมมติว่าการโจมตีของ Pokémon 1 ใช้ Attack กับ Defense ของ Pokémon 2
    attack_advantage = pokemon1_data['Attack'] / (pokemon2_data['Defense'] + 1)  # +1 เพื่อป้องกันการหารด้วย 0
    # สำหรับ Sp. Atk กับ Sp. Def
    special_attack_advantage = pokemon1_data['Sp. Atk'] / (pokemon2_data['Sp. Def'] + 1)
    
    # คำนวณความได้เปรียบรวม
    total_advantage = (attack_advantage + special_attack_advantage) * type_adv
    return total_advantage

# คำนวณ Speed Advantage
def calculate_speed_advantage(pokemon1_speed, pokemon2_speed):
    if pokemon1_speed > pokemon2_speed:
        return 1.0  # Pokémon 1 ได้โจมตีก่อน
    elif pokemon1_speed < pokemon2_speed:
        return 0.0  # Pokémon 2 ได้โจมตีก่อน
    else:
        return 0.5  # เสมอกัน
# คำนวณ Total Stat Advantage
def calculate_stat_advantage(pokemon1_total_stat, pokemon2_total_stat):
    if abs(pokemon1_total_stat - pokemon2_total_stat) > 200:
        if pokemon1_total_stat > pokemon2_total_stat:
            return 1  # Pokemon 1 wins
        else:
            return 0  # Pokemon 2 wins
    return None  # ไม่สามารถตัดสินได้จาก Total Stat

# ทำความสะอาดข้อมูล
def clean_data(data):
    data = data[['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    data['Type 2'] = data['Type 2'].fillna('None')
    data = data[~data['Name'].str.contains('Mega')]
    data['Total'] = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)
    return data

# สร้างโมเดล Neural Network
def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, kernel_regularizer=l2(0.001)),  # เพิ่มขนาด Layer
        LeakyReLU(alpha=0.3),  # ใช้ Leaky ReLU แทน ReLU
        BatchNormalization(),
        Dropout(0.5),  # เพิ่ม Dropout
        Dense(256, kernel_regularizer=l2(0.001)),  # เพิ่มขนาด Layer
        LeakyReLU(alpha=0.3),  # ใช้ Leaky ReLU แทน ReLU
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.3),  # ใช้ Leaky ReLU แทน ReLU
        BatchNormalization(),
        Dropout(0.4),
        Dense(64),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)  # เพิ่ม Learning Rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ทำนายผลการต่อสู้
# คำนวณผลการต่อสู้
def predict_battle_with_stat_advantage(model, pokemon1_data, pokemon2_data, type_advantage, speed_advantage, scaler):
    pokemon1_total_stat = pokemon1_data['Total']
    pokemon2_total_stat = pokemon2_data['Total']

    # คำนวณผลการต่อสู้จาก Total Stat
    stat_advantage = calculate_stat_advantage(pokemon1_total_stat, pokemon2_total_stat)

    # ถ้ามีการตัดสินจาก Total Stat โดยอัตโนมัติ
    if stat_advantage is not None:
        return stat_advantage  # 1 หรือ 0 ขึ้นอยู่กับว่าใครชนะจาก Total Stat

    # ถ้าไม่สามารถตัดสินจาก Total Stat ให้คำนวณจากข้อมูลอื่นๆ
    pokemon1_stats = pokemon1_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']].values
    pokemon2_stats = pokemon2_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']].values
    features = np.concatenate([pokemon1_stats, pokemon2_stats, [type_advantage], [speed_advantage]])

    # สเกลข้อมูล
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    
    return prediction[0][0]

# Streamlit App
def predict_pokemon():
    st.title("Pokemon Battle Predictor ⚔️")
    data = clean_data(load_data())
    type_effectiveness = load_type_effectiveness()
    pokemon_list = data['Name'].unique()

    pokemon1 = st.selectbox("Select Pokemon 1", pokemon_list)
    pokemon2 = st.selectbox("Select Pokemon 2", pokemon_list)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{pokemon1}**")
        image_url1 = get_pokemon_image_url(pokemon1)
        if image_url1:
            st.image(image_url1, width=150)
    with col2:
        st.write(f"**{pokemon2}**")
        image_url2 = get_pokemon_image_url(pokemon2)
        if image_url2:
            st.image(image_url2, width=150)

    if pokemon1 and pokemon2:
        pokemon1_data = data[data['Name'] == pokemon1].iloc[0]
        pokemon2_data = data[data['Name'] == pokemon2].iloc[0]
        type_advantage = calculate_type_advantage(
            pokemon1_data['Type 1'], pokemon1_data['Type 2'],
            pokemon2_data['Type 1'], pokemon2_data['Type 2'],
            type_effectiveness
        )
        speed_advantage = calculate_speed_advantage(pokemon1_data['Speed'], pokemon2_data['Speed'])

        data = clean_data(load_data())

        # สร้าง features X จากข้อมูลโปเกมอน
        X = pd.concat([data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']]] * 2, axis=1)
        X['Type Advantage'] = np.random.uniform(0.5, 2.0, len(X))  
        X['Speed Advantage'] = np.random.uniform(0.5, 1.5, len(X))  # เพิ่ม Speed Advantage ในฟีเจอร์

        # สร้าง Label y (1 = Pokémon 1 ชนะ, 0 = Pokémon 2 ชนะ)
        y = (X.iloc[:, :7].sum(axis=1) > X.iloc[:, 7:14].sum(axis=1)).astype(int)

        # Data Augmentation - สลับตำแหน่ง Pokémon
        X_aug = X.copy()
        X_aug.iloc[:, :7], X_aug.iloc[:, 7:14] = X.iloc[:, 7:14].values, X.iloc[:, :7].values
        y_aug = 1 - y
        X = pd.concat([X, X_aug])
        y = np.concatenate([y, y_aug])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        # สร้างโมเดลใหม่ทุกครั้ง
        model = build_model(X_train.shape[1])

        # เทรนโมเดล
        # เทรนโมเดล
        model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_data=(X_test, y_test), 
          callbacks=[early_stopping, lr_scheduler], verbose=2)

        # บันทึกโมเดลทุกครั้ง
        model.save("pokemon_model.h5")

        prediction = predict_battle_with_stat_advantage(model, pokemon1_data, pokemon2_data, type_advantage, speed_advantage, scaler)


        st.subheader("🎮 Battle Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{pokemon1}** Stats:")
            for stat in ['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
                st.write(f"{stat}: {pokemon1_data[stat]}")
        with col2:
            st.write(f"**{pokemon2}** Stats:")
            for stat in ['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
                st.write(f"{stat}: {pokemon2_data[stat]}")

        st.write(f"**Type Advantage:** {type_advantage:.2f}")
        st.write(f"**Speed Advantage:** {speed_advantage:.2f}")

        # แสดงผลลัพธ์
        if prediction == 1:
            st.success(f"{pokemon1} Wins!")
            st.error(f"{pokemon2} Loses!")
        else:
            st.success(f"{pokemon2} Wins!")
            st.error(f"{pokemon1} Loses!")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        train = model.evaluate(X_train, y_train, verbose=0)
        st.subheader("📊 Model Performance")
        st.write(f"✅ **Test Accuracy:** {accuracy:.2f}")
        st.write(f"✅ **Train Accuracy:** {train[1]:.2f}")
        st.write(f"✅ **Accuracy:** {accuracy:.2f}")
        st.write(f"✅ **Loss:** {loss:.4f}")


        

def main():
    tf.compat.v1.enable_eager_execution()
    page = st.sidebar.selectbox("Select a page", ["About Vgsales", "Vgsales", "About Pokemon", "Predic Battle"])
    if page == "About Vgsales":
        home()
    elif page == "Vgsales":
        vgsales()
    elif page == "About Pokemon":
        pokedex()
    elif page == "Predic Battle":
        predict_pokemon()

if __name__ == "__main__":
    main()





    
