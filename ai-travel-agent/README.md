# AI Travel Agent with Memory

A personal travel assistant that runs entirely on your own machine. It remembers each traveller's preferences between conversations, and it can look up live weather and real places to ground its answers.

Everything is free: local models through Ollama, and open data services that need no API keys.

## Features

- **Long-term memory per user.** Preferences such as diet, budget or interests are extracted with [Mem0](https://github.com/mem0ai/mem0), stored in [Qdrant](https://qdrant.tech), and recalled when they're relevant to a new question.
- **Relevance filtering.** Only memories that closely match the question (similarity score ≥ `MEMORY_MIN_SCORE`) reach the model, so unrelated preferences don't leak into answers.
- **"Memories used" panel.** Each answer shows which memories shaped it and how well they matched.
- **Memory management.** View and delete saved memories from the sidebar.
- **Tool calling.** The model can call three tools:
  - `geocode`: coordinates and country of a place ([Nominatim](https://nominatim.org) / OpenStreetMap)
  - `get_weather`: daily forecast up to 16 days ahead ([Open-Meteo](https://open-meteo.com))
  - `find_places`: real points of interest nearby ([Overpass API](https://overpass-api.de) / OpenStreetMap), in 7 categories: attractions, history, food, vegetarian, nightlife, nature, beaches
- **Streaming answers** with a "Looking things up…" status that shows which tools ran.
- **Resilient.** If memory, the network or a tool fails, the chat keeps working and explains what's missing.

## Tech stack

| Part | Technology |
|---|---|
| UI | Streamlit |
| Chat and memory model | Ollama, `llama3.2:3b` |
| Embeddings | Ollama, `nomic-embed-text` (768 dimensions) |
| Memory layer | Mem0 |
| Vector database | Qdrant |
| External data | Nominatim, Open-Meteo, Overpass API |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com)
- Docker (to run Qdrant)

## Setup

**1. Download the models**

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Check they're installed with `ollama list`.

**2. Start Qdrant**

```bash
docker run -d -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**3. Install the Python packages**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install streamlit ollama mem0ai qdrant-client
```

**4. Run the app**

```bash
streamlit run app.py
```

Open the link Streamlit prints (usually http://localhost:8501), enter a username in the sidebar and start chatting.

## Configuration

Settings are at the top of `travel_agent.py` and can be overridden with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `llama3.2:3b` | Model that answers the traveller (must support tool calling) |
| `MEMORY_MODEL` | `llama3.2:3b` | Model Mem0 uses to extract facts |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model for memory search |
| `MEMORY_MIN_SCORE` | `0.5` | Minimum similarity for a memory to be used |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server |


> Don't change `EMBED_MODEL` once you have saved memories: memories stored with one embedding model can't be searched with another.

## How it works

For each message:

1. **Recall.** Search Qdrant for the user's memories that match the question and keep only those above `MEMORY_MIN_SCORE`.
2. **Decide on tools.** If the question mentions things like weather, restaurants, places or "near", tools are offered to the model. General questions ("best time to visit Japan?") are answered directly, because small models tend to call tools whenever they're available.
3. **Use tools.** The model chooses which tools to call. The app runs them and sends the results back. This step isn't shown, so the model's in-between notes never appear in the chat.
4. **Answer.** The final answer is streamed, with tools switched off so the model has to answer rather than loop.
5. **Remember.** The user's message, plus the assistant message it was replying to (for context), goes to Mem0, which extracts short facts like "Is vegetarian". The new answer isn't sent, so the assistant's own suggestions aren't saved as the user's preferences.

## Test prompts

Send these in order with a fresh username:

| # | Prompt | What to check |
|---|---|---|
| 1 | `I prefer budget hotels and love museums.` | Sidebar memory shows both preferences |
| 2 | `Suggest a weekend city break.` | Uses both preferences; "Memories used" lists them |
| 3 | `What's the weather in Budapest this week?` | Weather tool runs; real temperatures |
| 4 | `What museums can I visit there?` | Understands "there"; `find_places` returns real museums |
| 5 | `Any cheap places to eat near the Hungarian National Museum?` | Food search near a landmark |
| 6 | `Do I need a visa to visit Hungary from Spain?` | No tools and no "Memories used" |
| 7 | `What's the best time of year to visit Japan?` | Answers directly, no tools |
| 8 | `What's the weather in Xyzzyville?` | Says the place couldn't be found, without crashing |

## Limitations

- **No authentication.** Anyone can type any username and see that user's memories. Fine for local use; add a login before deploying.
- **Chat history isn't saved.** Conversations reset when the page is refreshed (memories persist).
- **Small model.** `llama3.2:3b` sometimes skips a tool, picks the wrong one, or mixes up topics. A larger model improves this.
- **Keyword-based tool gate.** A question that needs a tool but uses none of the trigger words will be answered without tools. Add words to `TOOL_HINTS` if needed.
- **Forecasts cover 16 days.** Trips further out get general advice instead of a forecast.


## Data sources and credits

- Map and place data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), via Nominatim and the Overpass API
- Weather data by [Open-Meteo](https://open-meteo.com)