import streamlit as st
import requests
import json
from mistralai import Mistral
import time
from audio_recorder_streamlit  import audio_recorder
import speech_recognition as sr
from io import BytesIO
from create_db import create_and_insert_db

# Llamar a la función para crear la base de datos e insertar los datos
create_and_insert_db('customer_support_tickets.csv')
print("Base de datos inicializada y datos insertados.")

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

if 'selected_user' not in st.session_state:
    st.session_state['selected_user'] = None
    
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
        return f"Error: {response}"

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
    text = ""
    audio = audio_recorder('')
    if audio:
            st.audio(audio, format="audio/wav")
            audio_bytes = BytesIO(audio)
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_bytes) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="es-ES")
    
    return text

def show_users():
    st.session_state['selected_user'] = st.radio(
        "Select sesion user (Mock):",
        options=['Andres', 'CodeGPT','GuruSup', 'Mistral']  ,
        index=0
    )

# Interfaz principal
st.title('JustAsk')


with st.sidebar:
    if not st.session_state['CODEGPT_API_KEY']:
        st.session_state['CODEGPT_API_KEY'] = st.text_input(
            "CodeGPT API Key:",
            type="password"
        )
  
    if st.session_state['provider'] == 'CodeGPT' and st.session_state['CODEGPT_API_KEY']:
        agents = get_agents(st.session_state['CODEGPT_API_KEY'])
        st.session_state['selected_agent'] = st.selectbox(
            "Select an agent (CodeGPT):",
            options=agents,
            format_func=lambda agent: f"{agent['name']}"
        )
    
    show_users()
    
    if st.button("Reset Chat"):
        reset_chat()
        


def process_user_prompt(user_prompt: str):
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


# Chat interface
if (st.session_state['provider'] == 'CodeGPT' and st.session_state['selected_agent']) or \
    (st.session_state['provider'] == 'Mistral AI'):
        
    for message in st.session_state['messages']:
        with st.chat_message("user" if message['role'] == "user" else "assistant"):
            st.write(message['content'])

    text_audio = get_text_audio()
    user_prompt = text_audio
    
    process_user_prompt(user_prompt)
    
def run_swarms(user_prompt: str, num_iterations: int):
    """
    Ejecuta un proceso iterativo para generar y validar respuestas basadas en la intención del usuario.

    Args:
        user_prompt (str): El mensaje o pregunta inicial del usuario.
        num_iterations (int): El número actual de iteraciones del proceso.

    Returns:
        str: La respuesta generada si es válida o después de un número específico de iteraciones.
    """
    
    # Define la intención del agente basada en el mensaje del usuario.
    intetion = agent_intent_definition(user_prompt)
    
    # Obtiene los datos de contexto necesarios para la intención definida.
    context_data = agent_intent_definition(intetion)
    
    # Genera una respuesta basada en los datos de contexto.
    answer = agent_response_generation(context_data)
    
    # Si se ha alcanzado el número máximo de iteraciones (3), devuelve la respuesta generada.
    if num_iterations == 3:
        return answer
    
    # Valida la respuesta generada.
    answer_is_valid = agent_text_assistant_referee(answer)
    
    # Si la respuesta no es válida, refactoriza el mensaje del usuario y vuelve a ejecutar el proceso.
    if not answer_is_valid:
        new_prompt = agent_refactor_promp(user_prompt, answer)
        return run_swarms(new_prompt, num_iterations + 1)

def agent_intent_definition(user_prompt: str):
    prompt = f"""
    <prompt>
        <question>{user_prompt}</question>
        <context>Define the user's intention based on the provided prompt.</context>
        <expected_response>The intention of the user is clearly identified.</expected_response>
    </prompt>
    """
    return prompt

def agent_context_data_retrieval(intention_prompt: str):
    prompt = f"""
    <prompt>
        <question>{intention_prompt}</question>
        <context>Retrieve relevant context data for the identified intention.</context>
        <expected_response>Context data relevant to the user's intention is retrieved.</expected_response>
    </prompt>
    """
    return prompt

def agent_response_generation(user_prompt: str, context_data: str):
    prompt = f"""
    <prompt>
        <question>{user_prompt}</question>
        <context>{context_data}</context>
        <expected_response>A comprehensive response is generated based on the context data.</expected_response>
    </prompt>
    """
    return prompt

def agent_text_assistant_referee(answer: str):
    prompt = f"""
    <prompt>
        <question>{answer}</question>
        <context>Validate the generated response to ensure it meets the expected criteria.</context>
        <expected_response>The response is validated as correct or incorrect.</expected_response>
    </prompt>
    """
    return prompt

def agent_refactor_promp(user_prompt: str, answer: str):
    prompt = f"""
    <prompt>
        <question>{user_prompt}</question>
        <context>{answer}</context>
        <expected_response>The user prompt is refactored to improve the response generation.</expected_response>
    </prompt>
    """
    return prompt



