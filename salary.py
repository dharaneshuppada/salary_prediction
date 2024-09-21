import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and clean the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Salary_dataset.csv')
    data_clean = data.drop(columns=['Unnamed: 0'])
    return data_clean

# Train the model
@st.cache_resource
def train_model():
    data = load_data()
    X = data[['YearsExperience']]
    y = data['Salary']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Main Streamlit App
def main():
    st.title('Salary Prediction App')
    
    # Input for years of experience
    years_exp = st.number_input('Enter Years of Experience:', min_value=0.0, step=0.1)
    
    # Load and train model
    model = train_model()

    # Make prediction
    if st.button('Predict Salary'):
        prediction = model.predict([[years_exp]])[0]
        st.write(f'Predicted Salary for {years_exp} years of experience: ${prediction:,.2f}')
    
    # Display dataset
    if st.checkbox('Show Dataset'):
        data = load_data()
        st.write(data)

# Run the app
if __name__ == '__main__':
    main()
