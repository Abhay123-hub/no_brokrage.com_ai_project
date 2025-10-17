import streamlit as st
import requests
from datetime import date, time
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Get your property data here🌌")


question = st.text_input("enter your query")



# --- Submit ---
if st.button("Get Prediction"):
        inputs = {"user_query":question,"df_dict":{}}

        # Spinner while workflow executes
        with st.spinner("Getting response for you"):
            from workflow import Workflow
            work = Workflow()
            response = work.execute(inputs)

        st.success(response)