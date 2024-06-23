import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('decision_tree_opt.pkl', 'rb'))
loaded_scaler_model = pickle.load(open('scaler.pkl', 'rb'))

options_edu = {
    'Non-Graduate': 0,
    'Graduate': 1
}
options_emp ={
    'Employed': 1,
    'Unemployed': 0
}
class Loan:
    def __init__(self):
        st.title('LOAN APPROVAL PREDICTION')
        self.dependants = st.number_input('Enter number of dependant', min_value=0)
        selected_option_edu = st.selectbox('Select an option:', list(options_edu.keys()))
        self.education = options_edu[selected_option_edu]
        selected_option_emp = st.selectbox('Select an option:', list(options_emp.keys()))
        self.self_employed = options_emp[selected_option_emp]
        self.income = st.number_input('Annual Income', min_value=0)
        self.loan_amount= st.number_input('Loan Amount', min_value=0)
        self.loan_term = st.number_input('Loan Term', min_value=0)
        self.cibil = st.number_input('Credit Score', min_value=0)
        self.residential_assets_value = st.number_input('Residential Asset Value', min_value=0)
        self.commercial_assets_value = st.number_input('Commercial Asset Value', min_value=0)
        self.luxury_assets_value = st.number_input('Luxury Asset Value', min_value=0)
        self.bank_asset_value = st.number_input('Bank Asset Value', min_value=0)
        self.repayable= (self.bank_asset_value + self.luxury_assets_value + self.commercial_assets_value + self.residential_assets_value) - self.loan_term
        self.debt_income = self.debt_to_income_ratio(self.loan_amount,self.income)

            # code for Prediction
        pred = ''

        # creating a button for Prediction

        if st.button('PREDICT'):
            pred = self.loan_prediction([
            self.dependants,self.education,self.self_employed,self.income,self.loan_amount,self.loan_term,self.cibil,
            self.residential_assets_value,self.commercial_assets_value,self.luxury_assets_value,self.bank_asset_value,
            self.repayable,self.debt_income
        ])
            
        st.success(pred)
    def debt_to_income_ratio(self,x,y):
        if y != 0:
           return x / y
        else:
            return 0

    def loan_prediction(self, input_data):
        self.input_data = input_data
        # changing the input_data to numpy array
        self.input_data_as_numpy_array = np.asarray(self.input_data)

        # reshape the array as we are predicting for one instance
        self.input_data_reshaped = self.input_data_as_numpy_array.reshape(1,-1)
        self.input_scaled_data = loaded_scaler_model.transform(self.input_data_reshaped)
        self.prediction = loaded_model.predict(self.input_scaled_data)
        print(self.prediction)

        if (self.prediction == 0):
            return 'Rejected'
        else:
            return 'Accepted'

obj = Loan()



