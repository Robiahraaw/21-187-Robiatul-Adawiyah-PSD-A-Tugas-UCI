import streamlit as st
import pickle
import numpy as np

# Load model files
with open('scaler_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('pca_model.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

with open('knn_model.pkl', 'rb') as knn_file:
    knn_classifier = pickle.load(knn_file)

# Streamlit app
st.title('Prediksi Tingkat Tumor Otak')

# Input form for new data
st.write('Prediksi Tingkat Tumor Otak akan menghasilkan dua prediksi, yaitu Tumor Otak Ganas dan Tumor Otak Jinak.')
gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
gender = 0 if gender == 'Pria' else 1

age = st.number_input('Usia saat di diagnosa', min_value=0, max_value=100)

race = st.selectbox('Ras', ['Kulit Putih', 'Kulit Hitam', 'Asia', 'Alaska'])
race = ['Kulit Putih', 'Kulit Hitam', 'Asia', 'Alaska'].index(race)

mutation_values = ['Tidak Bermutasi', 'Bermutasi']

idh1 = st.selectbox('Mutasi Isocitrate Dehydrogenase', mutation_values)
idh1 = 0 if idh1 == 'Tidak Bermutasi' else 1

tp53 = st.selectbox('Mutasi Tumor Protein', mutation_values)
tp53 = 0 if tp53 == 'Tidak Bermutasi' else 1

atrx = st.selectbox('Mutasi ATRX Chromatin Remodeler', mutation_values)
atrx = 0 if atrx == 'Tidak Bermutasi' else 1

pten = st.selectbox('Mutasi Phosphatase and Tensin Homolog', mutation_values)
pten = 0 if pten == 'Tidak Bermutasi' else 1

egfr = st.selectbox('Mutasi Epidermal Growth Factor Receptor', mutation_values)
egfr = 0 if egfr == 'Tidak Bermutasi' else 1

cic = st.selectbox('Mutasi Capicua Transcriptional Repressor', mutation_values)
cic = 0 if cic == 'Tidak Bermutasi' else 1

muc16 = st.selectbox('Mutasi Mucin 16, Cell Surface Associated', mutation_values)
muc16 = 0 if muc16 == 'Tidak Bermutasi' else 1

pik3ca = st.selectbox('Mutasi Phosphatidylinositol-4,5-bisphosphate 3-kinase Catalytic Subunit Alpha', mutation_values)
pik3ca = 0 if pik3ca == 'Tidak Bermutasi' else 1

nf1 = st.selectbox('Mutasi Neurofibromin 1', mutation_values)
nf1 = 0 if nf1 == 'Tidak Bermutasi' else 1

pik3r1 = st.selectbox('Mutasi Phosphoinositide-3-kinase Regulatory Subunit 1', mutation_values)
pik3r1 = 0 if pik3r1 == 'Tidak Bermutasi' else 1

fubp1 = st.selectbox('Mutasi Far Upstream Element Binding Protein 1', mutation_values)
fubp1 = 0 if fubp1 == 'Tidak Bermutasi' else 1

rb1 = st.selectbox('Mutasi RB Transcriptional Corepressor 1', mutation_values)
rb1 = 0 if rb1 == 'Tidak Bermutasi' else 1

notch1 = st.selectbox('Mutasi Notch Receptor 1', mutation_values)
notch1 = 0 if notch1 == 'Tidak Bermutasi' else 1

bcor = st.selectbox('Mutasi BCL6 Corepressor', mutation_values)
bcor = 0 if bcor == 'Tidak Bermutasi' else 1

csmd3 = st.selectbox('Mutasi CUB and Sushi Multiple Domains 3', mutation_values)
csmd3 = 0 if csmd3 == 'Tidak Bermutasi' else 1

smarca4 = st.selectbox('Mutasi SWI/SNF Related, Matrix Associated, Actin Dependent Regulator of Chromatin, Subfamily a, Member 4', mutation_values)
smarca4 = 0 if smarca4 == 'Tidak Bermutasi' else 1

grin2a = st.selectbox('Mutasi Glutamate Ionotropic Receptor NMDA Type Subunit 2A', mutation_values)
grin2a = 0 if grin2a == 'Tidak Bermutasi' else 1

idh2 = st.selectbox('Mutasi Isocitrate Dehydrogenase (NADP(+)) 2', mutation_values)
idh2 = 0 if idh2 == 'Tidak Bermutasi' else 1

fat4 = st.selectbox('Mutasi FAT Atypical Cadherin 4', mutation_values)
fat4 = 0 if fat4 == 'Tidak Bermutasi' else 1

pdgfra = st.selectbox('Mutasi Platelet-Derived Growth Factor Receptor Alpha', mutation_values)
pdgfra = 0 if pdgfra == 'Tidak Bermutasi' else 1

if st.button('Prediksi Tingkat Tumor'):
    # Preprocess the input data
    new_data = np.array([gender, age, race, idh1, tp53, atrx, pten, egfr, cic, muc16, pik3ca, nf1, pik3r1, fubp1, rb1, notch1, bcor, csmd3, smarca4, grin2a, idh2, fat4, pdgfra]).reshape(1, -1)
    new_data = scaler.transform(new_data)
    new_data = pca.transform(new_data)

    # Make a prediction
    prediction = knn_classifier.predict(new_data)

    if prediction[0] == 0:
        st.write('Hasil Prediksi: Tumor Otak Jinak')
    else:
        st.write('Hasil Prediksi: Tumor Otak Ganas')