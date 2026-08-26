import streamlit as st

st.header('st.button')

# cara kerja st.button()
# argumen adalah display buttonnya, lalu nanti akan digantikan true atau false.

if st.button('Dari Jetson'):
    st.write('Hello there')
else: 
    st.write('halo Jetson')
