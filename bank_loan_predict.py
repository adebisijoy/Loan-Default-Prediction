import numpy as np
import pickle as pk
import streamlit as st

# loading the saved model
with open("predict_data.pkl", "rb") as f:  # Open in binary mode
    loaded_model = pk.load(f)


# Creating a function for Prediction


def loan_prediction(input_data):
    # Convert input data to float and handle any errors
    try:
        input_data_numeric = [float(val) for val in input_data]
    except ValueError:
        return "Please enter valid numeric values for all input"

    # Reshape the numpy array as we predict for one datapoint
    input_data_reshaped = np.array(input_data_numeric).reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The loan is repaid"
    else:
        return "The loan is defaulted"


def main():

    st.title("Bank Loan Predictor")

    # getting the input data from user
    loan = st.text_input("Loan")
    mortdue = st.text_input("Mortgage_due")
    value = st.text_input("Value")
    job = st.selectbox("Job", ["Other", "Office", "Sales", "Mgr", "ProfExe", "Self"])
    yoj = st.text_input("YOJ")
    derog = st.text_input("DEROG")
    delinq = st.text_input("Delinq")
    clage = st.text_input("Clage")
    ninq = st.text_input("Ninq")
    clno = st.text_input("Clno")
    debt_inc = st.text_input("Debt_Inc")
    debt_con = st.text_input("Debt_Con")
    home_imp = st.selectbox("Home Improvement", ["Yes", "No"])  # Changed to a selectbox for binary choice

    # code for prediction
    loan_def = ''

    # creating a button for prediction
    if st.button("Loan Predict Result"):
        # Converting job and home_imp to numeric values
        job_mapping = {"Other": 0, "Office": 1, "Sales": 2, "Mgr": 3, "ProfExe": 4, "Self": 5}
        home_imp_mapping = {"Yes": 1, "No": 0}
        job_val = job_mapping[job]
        home_imp_val = home_imp_mapping[home_imp]

        loan_def = loan_prediction([loan, mortdue, value, job_val, yoj, derog,
                                    delinq, clage, ninq, clno, debt_inc, debt_con, home_imp_val])

    st.success(loan_def)


if __name__ == '__main__':
    main()