import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , r2_score
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense , Input # type: ignore
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import StandardScaler
import requests 

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
    purpose = '''The purpose of this algorithm is to predict the outcome of a Pokémon battle based on various attributes of the Pokémon. It utilizes a machine learning model, specifically a Neural Network (Deep Learning), to learn the patterns and relationships in the battle data. This allows the model to predict which Pokémon is more likely to win based on their attributes, such as their statistics (HP, Attack, Defense, Speed, etc.) and type advantages.'''
    st.write(purpose)
    
    st.subheader("Problem Type")
    problem_type = '''
       Binary Classification Problem: The task is to predict which Pokémon is more likely to win in a battle based on their attributes.'''
    st.markdown(problem_type)

    st.subheader("Data Used")
    data_used = '''
        1.Pokémon images come from: :blue[https://pokeapi.co/.]\n
        2.:green[pokemon.csv] this datasets come from :blue[https://www.kaggle.com/datasets/abcsds/pokemon]\n
        This dataset contains :green[721 rows] and :green[9 columns] with various features related to Pokémon attributes and evolution. The features include:\n
        -**#** : A unique identifier for each Pokémon.\n
        -**Name** : The name of the Pokémon.\n
        -**Type 1** : The primary type of the Pokémon.\n
        -**Type 2** : The secondary type of the Pokémon.\n
        -**Total** : The total number of stats for the Pokémon.\n
        -**HP** : The hit points (HP) of the Pokémon.\n
        -**Attack** : The attack power of the Pokémon.\n
        -**Defense** : The defense power of the Pokémon.\n
        -**Sp. Atk** : The special attack power of the Pokémon.\n
        -**Sp. Def** : The special defense power of the Pokémon.\n
        -**Speed** : The speed of the Pokémon.\n
        -**Generation** : The generation in which the Pokémon was introduced.\n
        -**Legendary** : Indicates whether the Pokémon is legendary or not.\n
        3.:green[type_effectiveness.csv] The dataset comes from :blue[https://pokemondb.net/type]\n
        This dataset contains 3 features related to the strengths and weaknesses of each type:\n
        -**Attacking Type** : The type that is being attacked.\n
        -**Defending Type** : The type that is defending against the attack.\n
        -**Effectiveness** : The effectiveness of the attack on the defending type.\n
        4. :green[pokemon_battles_features.csv] This dataset was generated by a script (train.py) because the real win probability of each Pokémon pair cannot be directly obtained. The dataset was created under the following conditions:\n
         o	If Pokémon 1's total stats exceed Pokémon 2’s by more than 30%, Pokémon 1 is considered the winner, reflecting the fact that significant stat differences generally result in one Pokémon winning decisively.\n
         o	Speed Advantage: If one Pokémon has much higher speed, it will attack first.\n
         o	Type Effectiveness: The type advantage is calculated based on the effectiveness of one type against another.\n
         o	A sample of 60,000 pairs was randomly generated to ensure diversity and realism in the dataset.\n
           Features include:\n
         o	Total_p1: Total stats of Pokémon 1.\n
         o	Type1_p1: Primary type of Pokémon 1.\n
         o	Type2_p1: Secondary type of Pokémon 1.\n
         o	Speed_p1: Speed of Pokémon 1.\n
         o	Total_p2: Total stats of Pokémon 2.\n
         o	Type1_p2: Primary type of Pokémon 2.\n
         o	Type2_p2: Secondary type of Pokémon 2.\n
         o	Speed_p2: Speed of Pokémon 2.\n
         o	Winner: The winning Pokémon (1 for Pokémon 1, 0 for Pokémon 2).\n
         o	Win_Probability: The probability of Pokémon 1 winning the battle.\n
          '''
    st.markdown(data_used)
    st.subheader("Missing Values")
    st.markdown("There may be missing values in columns such as **Type 2**, which may not be applicable for all Pokémon.")
    st.subheader("Data Preparation")
    prepo = '''Before training the model, the relevant data needs to be prepared. The dataset used for training consists of information from Pokémon battle simulations, such as:\n
•	Numeric features: Total stats (Total_p1, Total_p2), Speed (Speed_p1, Speed_p2)\n
•	Categorical features: Types of the Pokémon (Type1_p1, Type2_p1, Type1_p2, Type2_p2)\n
Preprocessing:\n
•	One-hot Encoding is used to handle the categorical features like the Pokémon types. This technique converts each type into a binary vector that represents the presence or absence of that type. If a Pokémon has only one type, the missing secondary type is replaced with "Unknown.\n"
\n•	StandardScaler is applied to scale the numeric features so that all features have a similar scale, which improves the model's convergence during training.
'''
    st.markdown(prepo)
    st.subheader("Model Architerture")
    moar = '''A Neural Network (NN) is chosen for this task due to its capability to learn complex relationships between the features. The model consists of the following components:\n
•	Input Layer: Takes the input features, which include both numeric (Total, Speed) and encoded categorical data (Pokémon types).\n
•	Hidden Layers: The model contains three fully connected Dense layers with ReLU activation functions, which help in learning non-linear relationships between the input features and the target. The layers have 128, 64, and 32 units respectively. These layers are used to learn the underlying patterns in the data and the interactions between the different Pokémon attributes.\n
•	Dropout: While not explicitly stated in the provided code, Dropout is a regularization technique that can be added to reduce overfitting by randomly disabling some neurons during training. This helps the model generalize better to unseen data.\n
•	Output Layer: The model has a single unit in the output layer with a Sigmoid activation function, which produces a probability value between 0 and 1. This value represents the likelihood of Pokémon 1 winning the battle. If the output value is greater than 0.5, Pokémon 1 is predicted to win; otherwise, Pokémon 2 is predicted to win.\n
'''
    st.markdown(moar)
    st.subheader("Model Training")
    mt = '''The model is trained using a binary classification approach, where the goal is to predict the winner of the battle. The steps in training the model are as follows:\n
•	Loss Function: The model uses binary cross-entropy as the loss function, which is appropriate for binary classification problems. It measures the difference between the predicted probabilities and the actual outcomes (0 or 1).\n
•	Optimizer: The model uses the Adam optimizer, which is an efficient optimization algorithm that adapts the learning rate during training, helping the model converge faster.\n
•	Metrics: The accuracy of the model is tracked during training to evaluate its performance.\n
The training process involves splitting the dataset into a training set and a test set (80% for training and 20% for testing), scaling the features, and fitting the model to the training data. The model is then evaluated on the test set to determine its performance.\n
'''
    st.markdown(mt)
    st.subheader("Prediction")
    pdt = '''Once the model is trained, it can be used to predict the outcome of a Pokémon battle. Given two Pokémon, the following steps occur during prediction:\n
1.	Input Features: The input consists of the Pokémon’s stats (Total and Speed) and types. The types are encoded using the one-hot encoder, and the numeric features are scaled using the trained scaler.\n
2.	Model Prediction: The processed input features are passed through the trained neural network to get the predicted probability of Pokémon 1 winning.\n
3.	Winner Determination: If the predicted probability is greater than 0.5, Pokémon 1 is predicted to win; otherwise, Pokémon 2 is predicted to win.'''
    st.markdown(pdt)

    st.subheader("Model Evaluation")
    eva = '''The model's performance is evaluated using the following metrics:\n
    •	Accuracy: The percentage of correct predictions (how often the model predicts the correct winner).\n
    •	Loss: The binary cross-entropy loss, which gives an indication of how well the model is fitting the data.\n
    Both metrics are displayed in the user interface, allowing users to assess the model’s performance.'''
# ------------------------------------------
# โหลดข้อมูล
# ------------------------------------------
@st.cache_data
def load_pokemon_data():
    return pd.read_csv("pokemon.csv")

@st.cache_data
def load_battle_features():
    df = pd.read_csv("pokemon_battles_features.csv")
    if "Win_Probability" in df.columns:
        df = df.drop(columns=["Win_Probability"])
    return df

# ------------------------------------------
# ดึง URL รูปภาพจาก PokéAPI
# ------------------------------------------
def get_pokemon_image_url(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['sprites']['front_default']
    return None

# ------------------------------------------
# เตรียมข้อมูลเพื่อฝึกโมเดลจากไฟล์ pokemon_battles_features.csv
# ------------------------------------------
@st.cache_data
def prepare_training_data():
    battle_df = load_battle_features()
    # Numeric features: Total_p1, Speed_p1, Total_p2, Speed_p2
    numeric_features = battle_df[["Total_p1", "Speed_p1", "Total_p2", "Speed_p2"]].values
    # Categorical features: Type1_p1, Type2_p1, Type1_p2, Type2_p2
    categorical_cols = ["Type1_p1", "Type2_p1", "Type1_p2", "Type2_p2"]
    cat_data = battle_df[categorical_cols].fillna("Unknown").astype(str)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_features = encoder.fit_transform(cat_data)
    X = np.hstack((numeric_features, cat_features))
    y = battle_df["Winner"].values.astype(np.float32)
    return X, y, encoder

# ------------------------------------------
# สร้างโมเดล Neural Network
# ------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------------------
# ฟังก์ชันเทรนโมเดล (จะเรียกเมื่อกดปุ่ม "Train Model")
# ------------------------------------------
def train_model():
    X, y, enc = prepare_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mdl = build_model(X_train.shape[1])
    mdl.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    loss, acc = mdl.evaluate(X_test, y_test)
    st.write(f"🎯 Model Accuracy: {acc:.2%}")
    st.write(f"🎯 Model Loss: {loss:.4f}")
    return mdl, scaler, enc, acc, loss

# ------------------------------------------
# ตรวจสอบสถานะโมเดลใน session_state
# ------------------------------------------
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "model_obj" not in st.session_state:
    st.session_state.model_obj = None
if "scaler_obj" not in st.session_state:
    st.session_state.scaler_obj = None
if "encoder_obj" not in st.session_state:
    st.session_state.encoder_obj = None
if "model_accuracy" not in st.session_state:
    st.session_state.model_accuracy = None
if "model_loss" not in st.session_state:
    st.session_state.model_loss = None

# ------------------------------------------
# ส่วนของ UI ด้วย Streamlit
# ------------------------------------------
def predict_battle(pokemon1, pokemon2, encoder, model, scaler):
    progress_bar = st.progress(0)
    # ดึงข้อมูลของ Pokémon จาก pokemon_data
    pokemon_data = load_pokemon_data()
    p1 = pokemon_data[pokemon_data['Name'] == pokemon1].iloc[0]
    p2 = pokemon_data[pokemon_data['Name'] == pokemon2].iloc[0]
    # สร้างฟีเจอร์ตัวเลข: Total และ Speed ของแต่ละตัว
    numeric_features = np.array([[p1['Total'], p1['Speed'], p2['Total'], p2['Speed']]])
    cat_input = [[p1['Type 1'], p1['Type 2'], p2['Type 1'], p2['Type 2']]]
    cat_features = encoder.transform(cat_input)
    features = np.hstack((numeric_features, cat_features))
    progress_bar.progress(50)
    features = scaler.transform(features)
    progress_bar.progress(75)
    prediction = model.predict(features)[0][0]
    progress_bar.progress(100)
    return pokemon1 if prediction > 0.5 else pokemon2

def predict_pokemon():
    st.title("Pokemon Battle Predictor ⚔️")
    pokemon_data = load_pokemon_data()
    pokemon_list = pokemon_data['Name'].unique()
    # กรองไม่เอา Mega
    pokemon_list = [p for p in pokemon_list if 'Mega' not in p]
    pokemon1 = st.selectbox("Select Pokemon 1", pokemon_list)
    pokemon2 = st.selectbox("Select Pokemon 2", pokemon_list)
    
    col1, col2 = st.columns(2)
    st.subheader("🎮 Battle Prediction")
    
    with col1:
        img_url1 = get_pokemon_image_url(pokemon1)
        if img_url1:
            st.image(img_url1, caption=f"{pokemon1} Image", use_container_width=True)
        st.write(f"**{pokemon1}** Stats:")
        p1_data = pokemon_data[pokemon_data['Name'] == pokemon1].iloc[0]
        for stat in ['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
            st.write(f"{stat}: {p1_data[stat]}")
    
    with col2:
        img_url2 = get_pokemon_image_url(pokemon2)
        if img_url2:
            st.image(img_url2, caption=f"{pokemon2} Image", use_container_width=True)
        st.write(f"**{pokemon2}** Stats:")
        p2_data = pokemon_data[pokemon_data['Name'] == pokemon2].iloc[0]
        for stat in ['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
            st.write(f"{stat}: {p2_data[stat]}")
    # ถ้ายังไม่มีการเทรนโมเดล ให้แสดงปุ่ม Train Model
    if not st.session_state.model_trained:
        if st.button("Start Battle"):
            with st.spinner("Training model..."):
                mdl, scl, enc, acc, loss = train_model()
                st.session_state.model_obj = mdl
                st.session_state.scaler_obj = scl
                st.session_state.encoder_obj = enc
                st.session_state.model_accuracy = acc
                st.session_state.model_loss = loss
                st.session_state.model_trained = True
                st.success("Model training completed!")
    else:
        st.write("Model already trained.")
    
    # ถ้าโมเดลถูกเทรนแล้ว ให้แสดงปุ่ม Start Battle
    if st.session_state.model_trained:
        winner = predict_battle(pokemon1, pokemon2, st.session_state.encoder_obj, st.session_state.model_obj, st.session_state.scaler_obj)
        st.subheader(f"🏆 Winner: {winner}")
    
    st.subheader("Model Performance")
    if st.session_state.model_trained:
        st.write(f"🎯 Model Accuracy: {st.session_state.model_accuracy:.2%}")
        st.write(f"🎯 Model Loss: {st.session_state.model_loss:.4f}")
    else:
        st.write("Model not trained yet.")

        

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





    
