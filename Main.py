import streamlit as st
import subprocess
st.set_page_config(page_title="Main Launcher")
st.title("Main Launcher for Applications")
st.subheader("Choose an application to launch:")
bg_image = "background_image.jpeg"
st.image(bg_image, use_column_width=True)

# Function to run a script
def run_script(script_name):
    subprocess.run(['streamlit', 'run', script_name], check=True)

# Buttons to launch different applications
if st.button("Launch Chatbot"):
    run_script('chatbot.py')

if st.button("Launch PDF Question Answering"):
    run_script('pdf.py')

#if st.button("Launch Url Question Answering"): 
#    run_script('url.py')

if st.button("Launch Chatbot via image"): 
    run_script('vision.py')



