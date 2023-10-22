import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("""
    Data Mining Classification
    Aplikasi Diagnosis Penyakit Diabetes Mellitus  
    """)

    # pilih metode
    opsi_metode = ['Naive Bayes', 'KNN', 'SVM', 'Random Forest']
    metode_dipilih = st.selectbox('Pilih Metode Di List', options = opsi_metode)
    st.write('Metode Yang Dipilih Adalah', metode_dipilih)

    def input_user():
        # membagi kolom
        col1, col2 = st.columns(2)
        with col1:
            Usia = st.text_input('Input Nilai Umur')
        with col2:
            Glukosa = st.text_input('Input Nilai Glukosa')
        with col1:
            Tekanan_darah = st.text_input('Input Nilai Tekanan Darah ')
        with col2:
            Ketebalan_kulit = st.text_input('Input Nilai Ketebalan Kulit ')
        with col1:
            Insulin = st.text_input('Input Nilai Insulin ')
        with col2:
            BMI = st.text_input('Input Nilai BMI ')
        data={'Usia':Usia,
            'Glukosa': Glukosa,
            'Tekanan_darah': Tekanan_darah,
            'Ketebalan_kulit': Ketebalan_kulit,
            'Insulin': Insulin,
            'BMI' : BMI}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

    #metode yang dipilih
    diab_diagnosis = ''
    if metode_dipilih == 'Naive Bayes':
        diabetes_model_NB = pickle.load(open('modelNBC_over.sav', 'rb'))
        if st.button('Test Prediksi Diabetes'):
            diab_prediction = diabetes_model_NB.predict(inputan)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'Pasien terkena diabetes'
            else:
                diab_diagnosis = 'Pasien tidak terkena diabetes'
        st.success(diab_diagnosis)
    elif metode_dipilih == 'KNN':
        diabetes_model_KNN = pickle.load(open('modelKNN_over.sav', 'rb'))
        if st.button('Test Prediksi Diabetes'):
            diab_prediction = diabetes_model_KNN.predict(inputan)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'Pasien terkena diabetes'
            else:
                diab_diagnosis = 'Pasien tidak terkena diabetes'
        st.success(diab_diagnosis)
    elif metode_dipilih == 'SVM':
        diabetes_model_SVM = pickle.load(open('modelSVM_over.sav', 'rb'))
        if st.button('Test Prediksi Diabetes'):
            diab_prediction = diabetes_model_SVM.predict(inputan)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'Pasien terkena diabetes'
            else:
                diab_diagnosis = 'Pasien tidak terkena diabetes'
        st.success(diab_diagnosis)
    elif metode_dipilih == 'Random Forest':
        diabetes_model_RF = pickle.load(open('modelRF_over2.sav', 'rb'))
        if st.button('Test Prediksi Diabetes'):
            diab_prediction = diabetes_model_RF.predict(inputan)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'Pasien terkena diabetes'
            else:
                diab_diagnosis = 'Pasien tidak terkena diabetes'
        st.success(diab_diagnosis)
    
if __name__== '__main__':
    main()
    



