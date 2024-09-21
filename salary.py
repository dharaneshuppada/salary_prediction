import streamlit as st
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('salary_prediction_model.pkl')
    return model

# Main Streamlit App
def main():
    st.title('Salary Prediction App')

    # Input for years of experience
    years_exp = st.number_input('Enter Years of Experience:', min_value=0.0, step=0.1)

    # Load the model
    model = load_model()

    # Prediction
    if st.button('Predict Salary'):
        prediction = model.predict([[years_exp]])[0]
        st.write(f'Predicted Salary for {years_exp} years of experience: ${prediction:,.2f}')

# Run the app
if __name__ == '__main__':
    main()
