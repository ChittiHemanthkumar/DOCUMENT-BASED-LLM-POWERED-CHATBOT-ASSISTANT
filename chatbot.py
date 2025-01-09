
#  replicate_api = 'r8_Aj0w3bsHsRQiZxT2WcnCWYfqpfkkziw2rsFRW'
#  os.environ['REPLICATE_API_TOKEN'] = 'r8_Aj0w3bsHsRQiZxT2WcnCWYfqpfkkziw2rsFRW'
#  st.subheader('Models and parameters')
#  selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['gemma-2B'], 
# key='selected_model')
#  selected_model='Llama2-7B'
#  if selected_model == 'Llama2-7B':
#     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
#  elif selected_model == 'Llama2-13B':
#     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
# topic=st.sidebar.selectbox('Choose a Subuject', ['Operating Systems', 'Compiler Design','Computer Networks'], key='selected_model_1')
# temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
# max_length = st.sidebar.slider('max_length', min_value=32, max_value=256, value=120, step=8)
 
# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#  st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# # Display or clear chat messages
# for message in st.session_state.messages:
#  with st.chat_message(message["role"]):
#     st.write(message["content"])
# def clear_chat_history():
#  st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
# # Function for generating LLaMA2 response. Refactored from https://github.com/a16zinfra/llama2-chatbot
# def generate_llama2_response(prompt_input):
#  string_dialogue = f"You are a helpful assistant. You should respond only to {topic} 
# questions. You do not respond as 'User' or pretend to be 'User'. If the question is about 
# {topic}, you should return output with 'in {topic} ....'"
#  for dict_message in st.session_state.messages:
#     if dict_message["role"] == "user":
#         string_dialogue += "User: " + dict_message["content"] + "\n\n"
#     else:
#         string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
#         output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
#                                input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
#                                       "temperature":temperature, "max_length":max_length, 
#                                         "repetition_penalty":1})
#  return output
# if prompt := st.chat_input(disabled=not replicate_api):
#  st.session_state.messages.append({"role": "user", "content": prompt})
#  with st.chat_message("user"):
#     st.write(prompt)
# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#  with st.chat_message("assistant"):
#     with st.spinner("Thinking..."):
#         response = generate_llama2_response(prompt)
#     placeholder = st.empty()
#     full_response = ''
#     for item in response:
#         full_response += item
#         placeholder.markdown(full_response)
#     placeholder.markdown(full_response)
#  message = {"role": "assistant", "content": full_response}
#  st.session_state.messages.append(message)
import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="Chatbot")

# Replicate Credentials
with st.sidebar:
    
    replicate_api = 'r8_Aj0w3bsHsRQiZxT2WcnCWYfqpfkkziw2rsFRW'
    os.environ['REPLICATE_API_TOKEN'] = 'r8_Aj0w3bsHsRQiZxT2WcnCWYfqpfkkziw2rsFRW'

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['gemma-2B'], key='selected_model')
    selected_model='Llama2-7B'

    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    topic=st.sidebar.selectbox('Choose a Subject', ['Operating Systems', 'Compiler Design','Computer Networks'], key='selected_model_1')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=256, value=120, step=8)
    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = f"You are a helpful assistant. You should respond only to {topic} questions. You do not respond as 'User' or pretend to be 'User'. If the question is about {topic}, you should return output with 'in {topic} ....'"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)