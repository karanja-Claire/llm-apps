# Local ChatGPT with Memory

A simple conversational AI application built with Python, Streamlit, and Ollama. It allows users to chat with a locally running LLM while maintaining short-term conversation history and long-term memory.

## Features

- **Local LLM integration:** Uses Ollama to run an AI model locally.
- **Conversational chat:** Allows users to interact with the AI through a Streamlit interface.
- **Short-term memory:** Maintains recent conversation history during the session.
- **Long-term memory:** Saves user facts in a local `memory.json` file.
- **Custom memory:** Use `/add <fact>` to teach the assistant something about you.
- **Streaming responses:** Displays AI responses as they are generated.
- **Temperature control:** Adjusts the creativity and variability of responses.
- **Memory management:** Clear saved memories or conversation history through the sidebar.

## Technologies Used

- Python
- Streamlit
- Ollama
- OpenAI Python SDK
- JSON

## Requirements

- Python 3.9+
- Ollama installed and running
- A locally downloaded Ollama model (default: `llama3.2:3b`)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Install dependencies

```bash
pip install streamlit openai
```

### 3. Download the Ollama model

```bash
ollama pull llama3.2:3b
```

### 4. Start Ollama

Make sure Ollama is running locally.

### 5. Run the application

```bash
streamlit run app.py
```


## Using the Application

- Type a message in the chat input to start a conversation.
- Use `/add <fact>` to save information to long-term memory.

Example:

```text
/add I enjoy travelling and exploring new cities.
```

- Use the sidebar to view saved memories.
- Adjust the temperature slider to change response variability.
- Use **Clear memory** to delete saved facts.
- Use **Clear chat** to clear the current conversation history.

## Environment Variables (Optional)

The application supports environment variables to configure the model connection:

| Variable | Default value |
|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` |
| `LLM_API_KEY` | `ollama` |
| `LLM_MODEL` | `llama3.2:3b` |

## Project Structure

```text
local-chatbot/
│
├── app.py
├── memory.json       # Created when memories are saved
└── README.md
```

## How Memory Works

The application uses two types of memory:

**1. Short-term memory**

Stores recent user and assistant messages in Streamlit session state. The latest 20 messages are included in each model request.

**2. Long-term memory**

Stores user-provided facts in a local JSON file. These facts are included in the system prompt so the assistant can use them in future conversations.

## Future Improvements

- Add user authentication and separate memories for different users.
- Use a vector database for semantic memory retrieval.
- Add memory editing and deletion for individual facts.
- Improve conversation history management.
- Add support for multiple local LLM models.

---
