import streamlit as st
import requests
import json
from mistralai import Mistral
import time
from audio_recorder_streamlit  import audio_recorder
import speech_recognition as sr
from io import BytesIO

# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)

# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    for r in response:
        if hasattr(r, 'delta') and r.delta.content:
            content = r.delta.content
            content = content.replace("$", "\$")
            yield content

# MAIN
st.set_page_config(
    page_title="LLMHackathon",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Función para reiniciar el chat
def reset_chat():
    st.session_state['messages'] = []

# Variables de estado
if 'CODEGPT_API_KEY' not in st.session_state:
    st.session_state['CODEGPT_API_KEY'] = st.secrets["CODEGPT_API_KEY"] if "CODEGPT_API_KEY" in st.secrets else ""

if 'MISTRAL_API_KEY' not in st.session_state:
    st.session_state['MISTRAL_API_KEY'] = st.secrets["MISTRAL_API_KEY"] if "MISTRAL_API_KEY" in st.secrets else ""

if 'MISTRAL_MODEL' not in st.session_state:
    st.session_state['MISTRAL_MODEL'] = st.secrets["MISTRAL_MODEL"] if "MISTRAL_MODEL" in st.secrets else "mistral-large-latest"

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'selected_agent' not in st.session_state:
    st.session_state['selected_agent'] = None

if 'provider' not in st.session_state:
    st.session_state['provider'] = 'Mistral AI'

# Funciones
def get_headers(auth_token):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

def get_agents(auth_token):
    headers = get_headers(auth_token)
    response = requests.get(
        url=f"{st.secrets['CODEGPT_API_URL']}agents",
        headers=headers
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to retrieve agents. Status code: {response.status_code}")
        return []

def get_agent_completion(agent_id: str, custom_content: str):
    headers = get_headers(st.session_state['CODEGPT_API_KEY'])
    messages = [{"role": "user", "content": custom_content}]
    payload = {
        "stream": False,
        "format": "text",
        "agentId": agent_id,
        "messages": messages
    }

    response = requests.post(
        headers=headers,
        url=f"{st.secrets['CODEGPT_API_URL']}chat/completions",
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}"

def get_mistral_completion(prompt: str):
    client = Mistral(api_key=st.session_state['MISTRAL_API_KEY'])
    chat_response = client.chat.complete(
        model=st.session_state['MISTRAL_MODEL'],
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content

def get_text_audio():
    text = None
    audio = audio_recorder()
    if audio:
            st.audio(audio, format="audio/wav")
            
            audio_bytes = BytesIO(audio)
            
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_bytes) as source:
                audio_data = recognizer.record(source)
            
                if not audio_data.frame_data.strip(b'\x00'):
                    return None
                
                text = recognizer.recognize_google(audio_data, language="es-ES")
    
    return text
    
# Interfaz principal
st.title('JustAsk')
if st.button("Reset Chat"):
    reset_chat()

with st.sidebar:
    if not st.session_state['CODEGPT_API_KEY']:
        st.session_state['CODEGPT_API_KEY'] = st.text_input(
            "CodeGPT API Key:",
            type="password"
        )
    # Si tenemos las credenciales de Mistral, mostramos la opción de seleccionar proveedor
    if st.session_state['MISTRAL_API_KEY'] and st.session_state['MISTRAL_MODEL']:
        st.session_state['provider'] = st.radio(
            "Select Provider:",
            options=['Mistral AI', 'CodeGPT'],
            index=0
        )
    if st.session_state['provider'] == 'CodeGPT' and st.session_state['CODEGPT_API_KEY']:
        agents = get_agents(st.session_state['CODEGPT_API_KEY'])
        st.session_state['selected_agent'] = st.selectbox(
            "Select an agent:",
            options=agents,
            format_func=lambda agent: f"{agent['name']}"
        )

# Chat interface
if (st.session_state['provider'] == 'CodeGPT' and st.session_state['selected_agent']) or \
    (st.session_state['provider'] == 'Mistral AI'):
    for message in st.session_state['messages']:
        with st.chat_message("user" if message['role'] == "user" else "assistant"):
            st.write(message['content'])

    text_audio = get_text_audio()
   
    if text_audio:
        user_prompt = st.chat_input("Say something")
        st.write("No se grabó ningún audio")
    else:
        user_prompt = text_audio

    if user_prompt:
        # Mostrar mensaje del usuario
        st.session_state['messages'].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Obtener y mostrar respuesta según el proveedor seleccionado
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if st.session_state['provider'] == 'CodeGPT':
                response = get_agent_completion(st.session_state['selected_agent']['id'], user_prompt)
                
                
                # Simular streaming para CodeGPT
                for chunk in stream_str(response):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            else:  # Mistral
                response = get_mistral_completion(user_prompt)
                # Simular streaming
                for chunk in stream_str(response):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state['messages'].append({"role": "assistant", "content": full_response})

    