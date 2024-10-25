# Crea una app con la API de Mistral y Code GPT

### Este proyecto implementa una aplicación de chat utilizando la API de Mistral y CodeGPT, desarrollada con [Streamlit](https://streamlit.io/).
Desarrollado por [@GustavoEspindola – CodeGPT](https://www.linkedin.com/in/gustavoespindola/)

<!-- add an image -->
<img src="https://raw.githubusercontent.com/gustavoespindola/llmhackathon-py/refs/heads/master/hackathon.gif" alt="llmhackathon" width="100%">

Este proyecto demuestra la integración de modelos avanzados de IA a través de:
- API de Mistral AI para procesamiento del lenguaje
- API de CodeGPT para interactuar con agentes especializados
- Streamlit para una interfaz de usuario intuitiva

## 🛠️ Requisitos Previos

Para utilizar esta aplicación necesitarás:

### Para la Implementación Básica
**Integración con Mistral API**
   - Obtener tu [Clave API de Mistral](https://console.mistral.ai/api-keys/)
   - Probar endpoints de la API usando el archivo .http proporcionado ([Documentación de Endpoints Mistral](https://docs.mistral.ai/api/#tag/models))
   - Implementar la funcionalidad de chat mediante Streamlit

### Para Funciones Avanzadas
**Configuración del Agente CodeGPT**
   - Crear un Agente AI a través de CodeGPT
   - Cargar los datos necesarios del agente
   - Configurar el ID del agente
   - Configurar la clave API de CodeGPT
   - Implementar interacciones con el agente en la interfaz de chat


## 📚 Recursos Esenciales

- [🎯 LLMHackathon](https://llmhackathon.dev/)
- [🔑 Documentación API Mistral](https://console.mistral.ai/api-keys/)
- [🎯 Mistral Endpoints](https://docs.mistral.ai/api/#tag/models)
- [📚 Documentación de Streamlit](https://streamlit.io/)
- [🚀 Registro en CodeGPT](app.codegpt.co/r/gustavo)
- [📖 API Key de CodeGPT](https://app.codegpt.co/en/apikeys)
- [📖 CodeGPT Documentation](https://developers.codegpt.co/reference/completion-beta)

---

## 🔧 Instrucciones de Instalación

**Clonar el repositorio**

`git clone https://github.com/gustavoespindola/llmhackathon-py`

**Ingresar a la carpeta del proyecto**

`cd llmhackathon-py`


**Instalar dependencias**
```bash
# Instalar dependencias requeridas
  pip install -r requirements.txt
```

**Iniciar el servidor de desarrollo**

```bash
  streamlit run app.py
```

``` bash
  You can now view your Streamlit app in your browser.

  ➜  Local URL: http://localhost:8501
  ➜  Network URL: http://192.168.100.5:8501
```

La aplicación estará disponible en `http://localhost:8501`

---

## 🌐 Guía de Despliegue

Despliegue en Streamlit Cloud

1. Crear cuenta en Streamlit
2. Crear nuevo proyecto
3. Vincular con el repositorio
4. Configurar variables de entorno
5. Ejecutar el despliegue

### Configurar variables de entorno:

- Manejo de secrets en [Streamlit Local](https://docs.streamlit.io/develop/api-reference/connections/st.secrets)
- Manejo de screts en [Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

Ingresa a la carpeta .secrets y abre el archivo secrets.toml. En este archivo, ingresa las siguientes variables de entorno:

Para ejemplo con Mistral utiliza la varible `MISTRAL_API_KEY` y `MISTRAL_MODEL`
[🔑 Documentación API Mistral](https://console.mistral.ai/api-keys/)

Para ejemplo con CodeGPT utiliza la varible `CODEGPT_API_KEY`
[📖 API Key de CodeGPT](https://app.codegpt.co/en/apikeys)


### Configurar variables de entorno:

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! No dudes en enviar un Pull Request.

## 📄 Licencia
Este proyecto está licenciado bajo la Licencia MIT.
