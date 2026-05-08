import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('melb_data.csv')
    
    # Filling missing values as in the notebook
    df["Car"].fillna(df["Car"].mode()[0], inplace=True)
    df["BuildingArea"] = df.groupby("Rooms")["BuildingArea"].transform(
        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else df["BuildingArea"].median())
    )
    df.loc[df["BuildingArea"] > df["Landsize"], "BuildingArea"] = df["Landsize"]
    df["YearBuilt"] = df.groupby("Suburb")["YearBuilt"].transform(
        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else df["YearBuilt"].median())
    )
    df["CouncilArea"] = df.groupby("Suburb")["CouncilArea"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df["CouncilArea"].mode()[0])
    )
    
    # Drop rows with missing Price
    df = df.dropna(subset=['Price'])
    return df

df = load_data()

# Prepare features
X = df.drop("Price", axis=1)
Y = df["Price"]
categorical_columns = X.select_dtypes(include=['object']).columns

# Fit encoders
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
@st.cache_resource
def train_model():
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_scaled, Y)
    return model

model = train_model()

# Streamlit app
st.title("Melbourne Housing Price Predictor")
st.write("Enter the property details to predict the price using XGBoost model.")

st.sidebar.header("Property Features")

# Input features
input_data = {}
input_data['Rooms'] = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)

# Map for Type
type_map = {'House': 'h', 'Unit': 'u', 'Townhouse': 't'}
selected_type = st.sidebar.selectbox("Property Type", list(type_map.keys()))
input_data['Type'] = type_map[selected_type]

input_data['Distance'] = st.sidebar.number_input("Distance to CBD (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
input_data['Landsize'] = st.sidebar.number_input("Land Size (sqm)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
input_data['BuildingArea'] = st.sidebar.number_input("Building Area (sqm)", min_value=0.0, max_value=input_data['Landsize'], value=min(100.0, input_data['Landsize']), step=10.0)
input_data['YearBuilt'] = st.sidebar.number_input("Year Built", min_value=1800, max_value=2023, value=2000, step=1)
input_data['Regionname'] = st.sidebar.selectbox("Region Name", sorted(df['Regionname'].unique()))

# Fill other columns with defaults
for col in X.columns:
    if col not in input_data:
        if col in categorical_columns:
            input_data[col] = df[col].mode()[0]
        else:
            input_data[col] = df[col].mean()

# Create input DataFrame
input_df = pd.DataFrame([input_data])

# Ensure column order matches training data
input_df = input_df[X.columns]

# Encode categorical
for col in categorical_columns:
    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug info:")
        st.write("Input data:", input_data)
        st.write("Input DF shape:", input_df.shape)
        st.write("Input scaled shape:", input_scaled.shape)