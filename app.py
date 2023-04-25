import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict sales
def predict_sales(features):
    # Convert input into dataframe
    input_df = pd.DataFrame([features])

    # Make predictions
    prediction = model.predict(input_df)

    return prediction[0]

# Create a web app
def main():
    st.title('Big Mart Sales Prediction')

    # Get input features from user
    item_weight = st.number_input('Item Weight')
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular', 'Non-Edible'])
    item_visibility = st.number_input('Item Visibility')
    item_type = st.selectbox('Item Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
                                           'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
                                           'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
                                           'Breads', 'Starchy Foods', 'Others'])
    item_mrp = st.number_input('Item MRP')
    outlet_establishment_year = st.number_input('Outlet Establishment Year')
    outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2',
                                                'Supermarket Type3', 'Grocery Store'])

    # Map input features to a dictionary
    input_dict = {'Item_Weight': item_weight,
                  'Item_Fat_Content': item_fat_content,
                  'Item_Visibility': item_visibility,
                  'Item_Type': item_type,
                  'Item_MRP': item_mrp,
                  'Outlet_Establishment_Year': outlet_establishment_year,
                  'Outlet_Size': outlet_size,
                  'Outlet_Location_Type': outlet_location_type,
                  'Outlet_Type': outlet_type}

    # Predict sales using the trained model
    if st.button('Predict Sales'):
        sales = predict_sales(input_dict)
        st.success(f'Predicted Sales: {sales:.2f}')

if __name__ == '__main__':
    main()
