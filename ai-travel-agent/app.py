import json
import math
import os
import textwrap
import time
from datetime import date

import ollama
import requests
import streamlit as st
from mem0 import Memory

# 0. Settings & Env Variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# llama3.2:3b is free, runs locally and supports tool calling.
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")       # answers the traveller
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "llama3.2:3b")   # extracts facts for Mem0
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
EMBED_DIMS = 768                                          # matches nomic-embed-text

MEMORY_SEARCH_LIMIT = 5
# Search always returns the 5 closest memories
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.5"))
HISTORY_LENGTH = 6

# Nominatim and Overpass ask apps to identify themselves.
HTTP_HEADERS = {"User-Agent": "ai-travel-agent-demo/1.0 (personal project)"}
HTTP_TIMEOUT = 20

# 1. Tools the model can call (free, no API keys)
#    - Geocoding: Nominatim (OpenStreetMap)
#    - Weather:   Open-Meteo
#    - Places:    Overpass API (OpenStreetMap)


_geocode_cache = {}
_last_nominatim_call = 0.0


def geocode(place: str) -> dict:
    """Find the coordinates and country of a city, town or landmark.

    Args:
        place: Name of the place, e.g. "Porto, Portugal" or "Eiffel Tower".

    Returns:
        Name, latitude, longitude and country of the best match.
    """
    key = place.strip().lower()
    if key in _geocode_cache:
        return _geocode_cache[key]

    # Nominatim's usage policy: at most one request per second.
    global _last_nominatim_call
    wait = 1.0 - (time.monotonic() - _last_nominatim_call)
    if wait > 0:
        time.sleep(wait)
    _last_nominatim_call = time.monotonic()

    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return {"error": f"Could not find a place called '{place}'."}

    top = data[0]
    result = {
        "name": top.get("display_name", place),
        "latitude": float(top["lat"]),
        "longitude": float(top["lon"]),
        "country": top.get("address", {}).get("country", ""),
    }
    _geocode_cache[key] = result
    return result


WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def get_weather(place: str, days: int = 7) -> dict:
    """Get the daily weather forecast for a place for the next 1-16 days.

    Only for upcoming dates. Do NOT use for general questions like "best time
    of year to visit" or typical climate: answer those from your own knowledge.

    Args:
        place: City or town name, e.g. "Seville".
        days: Number of days to forecast, from 1 to 16.

    Returns:
        Daily min/max temperature in Celsius, chance of rain and conditions.
    """
    loc = geocode(place)
    if "error" in loc:
        return loc
    days = max(1, min(int(days), 16))

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    d = resp.json().get("daily", {})

    forecast = [
        {
            "date": d["time"][i],
            "conditions": WEATHER_CODES.get(d["weather_code"][i], "unknown"),
            "max_c": d["temperature_2m_max"][i],
            "min_c": d["temperature_2m_min"][i],
            "rain_chance_pct": d["precipitation_probability_max"][i],
        }
        for i in range(len(d.get("time", [])))
    ]
    return {"place": loc["name"], "forecast": forecast}


PLACE_CATEGORIES = {
    "attractions": '["tourism"~"^(attraction|museum|gallery|viewpoint|zoo|aquarium|theme_park)$"]',
    "history": '["historic"~"^(castle|monument|ruins|memorial|archaeological_site|church)$"]',
    "food": '["amenity"~"^(restaurant|cafe)$"]',
    "vegetarian": '["amenity"~"^(restaurant|cafe)$"]["diet:vegetarian"~"^(yes|only)$"]',
    "nightlife": '["amenity"~"^(bar|pub|nightclub)$"]',
    "nature": '["leisure"~"^(park|nature_reserve|garden)$"]',
    "beaches": '["natural"="beach"]',
}


def _distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def find_places(place: str, category: str = "attractions", radius_km: float = 3) -> dict:
    """Find real points of interest near a place using OpenStreetMap.

    Args:
        place: City, neighbourhood or landmark to search around, e.g. "Bilbao".
        category: One of: attractions, history, food, vegetarian, nightlife, nature, beaches.
        radius_km: Search radius in kilometres, from 0.5 to 10.

    Returns:
        Up to 12 named places with their type and distance from the centre.
    """
    if category not in PLACE_CATEGORIES:
        return {"error": f"Unknown category. Use one of: {', '.join(PLACE_CATEGORIES)}"}
    loc = geocode(place)
    if "error" in loc:
        return loc
    lat, lon = loc["latitude"], loc["longitude"]
    radius_m = int(max(0.5, min(float(radius_km), 10)) * 1000)

    query = (
        "[out:json][timeout:25];"
        f"nwr{PLACE_CATEGORIES[category]}[name](around:{radius_m},{lat},{lon});"
        "out center 60;"
    )
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()

    places = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        plat = el.get("lat") or el.get("center", {}).get("lat")
        plon = el.get("lon") or el.get("center", {}).get("lon")
        if plat is None or plon is None:
            continue
        place_info = {
            "name": tags.get("name:en") or tags["name"],
            "type": tags.get("tourism") or tags.get("historic") or tags.get("amenity")
                    or tags.get("leisure") or tags.get("natural") or "",
            "distance_km": round(_distance_km(lat, lon, plat, plon), 1),
        }
        if tags.get("cuisine"):
            place_info["cuisine"] = tags["cuisine"]
        # Places with a Wikipedia page or website tend to be the notable ones.
        notability = ("wikipedia" in tags) * 2 + ("website" in tags)
        places.append((notability, place_info))

    places.sort(key=lambda p: (-p[0], p[1]["distance_km"]))
    return {"near": loc["name"], "category": category, "places": [p for _, p in places[:12]]}


TOOLS = [geocode, get_weather, find_places]
TOOLS_BY_NAME = {f.__name__: f for f in TOOLS}

TOOL_LABELS = {
    "geocode": "Locating {place}",
    "get_weather": "Checking the weather in {place}",
    "find_places": "Finding {category} near {place}",
}


def run_tool(name: str, arguments: dict) -> dict:
    """Run a tool call from the model. Every failure becomes an error the model can read."""
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except requests.RequestException as e:
        return {"error": f"{name} could not reach its data source: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}


def tool_label(name: str, args: dict) -> str:
    try:
        return TOOL_LABELS[name].format(**{"category": "places", **args})
    except (KeyError, IndexError):
        return f"Using {name}"


# Small models call tools for everything if tools are on offer, so we only
# offer them when the question looks like it needs live or local data.

TOOL_HINTS = [
    "weather", "forecast", "rain", "temperature", "sunny", "snow", "wind",
    "near", "nearby", "around", "restaurant", "eat", "food", "cafe", "café",
    "vegetarian", "vegan", "bar", "pub", "club", "jazz", "live music", "drinks",
    "nightlife", "night", "go out", "going out", "museum", "attraction",
    "things to do", "what to see", "what to do", "places", "sights", "park",
    "beach", "historic", "castle", "where is", "coordinates", "located",
]


def needs_tools(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in TOOL_HINTS)


def chat_with_tools(messages: list, use_tools: bool = True, on_tool=None):
    """Yield the answer text, so it can be passed straight to st.write_stream.

    1. If tools are allowed, ask the model once (not streamed) whether it wants any.
       Not streaming this step means its in-between notes ("let me try again")
       are never shown to the user.
    2. If it asked for tools, run them all, then stream the final answer with
       tools switched off, so it has to answer rather than loop.
    """
    if use_tools:
        response = ollama_client.chat(model=CHAT_MODEL, messages=messages, tools=TOOLS)
        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            yield response.message.content or ""
            return

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": tc.function.name, "arguments": dict(tc.function.arguments)}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            name, args = tc.function.name, dict(tc.function.arguments)
            result = run_tool(name, args)
            if on_tool:
                on_tool(name, args, result)
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    stream = ollama_client.chat(model=CHAT_MODEL, messages=messages, stream=True)
    for chunk in stream:
        if chunk.message.content:
            yield chunk.message.content


# 2. Configure the application

st.set_page_config(page_title="AI Travel Agent")

st.title("AI Travel Agent with Memory")
st.caption("Your personal travel assistant that remembers your preferences.")

# 3. Configure Mem0 and Qdrant


@st.cache_resource
def initialize_memory():
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "travel_memories",
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "embedding_model_dims": EMBED_DIMS,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": MEMORY_MODEL,
                "temperature": 0,
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBED_MODEL,
                "ollama_base_url": OLLAMA_URL,
            },
        },
    }
    return Memory.from_config(config)


memory = initialize_memory()
ollama_client = ollama.Client(host=OLLAMA_URL)

# 4. User identification

st.sidebar.title("Traveller Profile")

user_id = st.sidebar.text_input("Enter your username", value="traveller_001").strip()

if "active_user" not in st.session_state:
    st.session_state.active_user = user_id

if user_id != st.session_state.active_user:
    st.session_state.messages = []
    st.session_state.active_user = user_id

if "messages" not in st.session_state:
    st.session_state.messages = []

if not user_id:
    st.sidebar.warning("Please enter a username to start chatting.")

# 5. View and manage saved memories

st.sidebar.divider()

with st.sidebar.expander("Your Long-Term Memory"):
    if not user_id:
        st.info("Enter a username to see saved memories.")
    else:
        try:
            saved = memory.get_all(filters={"user_id": user_id}).get("results", [])
        except Exception as e:
            saved = []
            st.error(f"Couldn't load memories: {e}")

        if not saved:
            st.info("No memories found.")

        for item in saved:
            col1, col2 = st.columns([5, 1])
            col1.write(item.get("memory", ""))
            if col2.button("🗑️", key=f"delete_{item['id']}", help="Delete this memory"):
                memory.delete(item["id"])
                st.rerun()

# 6. Display conversation history

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Accept user input

prompt = st.chat_input("Where would you like to travel?", disabled=not user_id)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

# 8. Retrieve relevant memories (optional: chat still works without them)

    relevant = []
    try:
        results = memory.search(
            query=prompt,
            filters={"user_id": user_id},
            limit=MEMORY_SEARCH_LIMIT,
        ).get("results", [])
        # Drop weak matches so unrelated memories don't leak into the answer.
        relevant = [r for r in results if (r.get("score") or 0) >= MEMORY_MIN_SCORE]
    except Exception as e:
        st.toast(f"Memory unavailable, answering without it ({e})")
    context = "\n".join(f"- {r['memory']}" for r in relevant)

# 9. Prepare LLM context

    system_prompt = textwrap.dedent(f"""\
        You are a friendly travel assistant. Today is {date.today():%A %d %B %Y}.

        Traveller's saved preferences:
        {context or "None relevant to this question."}


        Rules:
        - Answer only the traveller's latest message. Earlier messages are
          background; don't bring up old topics unless asked.
        - If the latest message names a place, answer about that place, even if
          earlier messages or saved preferences mention a different one. The
          traveller can change destination at any time; never tell them they
          meant somewhere else.
        - Use the saved preferences only if they matter for this question.
          Never invent preferences.
        - If tool results are provided, base your answer on them and don't
          make up forecasts or places.
        - Be concise.
    """)

    conversation = [{"role": "system", "content": system_prompt}]
    conversation += [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-HISTORY_LENGTH:]
    ]

# 10. Generate response (streamed, with tool calls in between)

    with st.chat_message("assistant"):
        status_slot = st.empty()
        status = {}

        def show_tool(name, args, result):
            # Create the "Looking things up" box the first time a tool is used.
            if "box" not in status:
                status["box"] = status_slot.status("Looking things up…", expanded=False)
            mark = "⚠️" if "error" in result else "✅"
            status["box"].write(f"{mark} {tool_label(name, args)}")

        try:
            answer = st.write_stream(
                chat_with_tools(conversation, use_tools=needs_tools(prompt), on_tool=show_tool)
            )
        except Exception as e:
            st.error(f"Couldn't generate a response: {e}")
            st.stop()

        if "box" in status:
            status["box"].update(label="Looked things up", state="complete")

        # Shows which memories shaped this answer (handy for spotting bad ones).
        if relevant:
            with st.expander("Memories used"):
                for r in relevant:
                    st.caption(f"{r['memory']}  (match {r.get('score', 0):.2f})")

    if not answer:
        answer = "Sorry, I couldn't come up with an answer. Could you rephrase?"
        st.warning(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 11. Save user information in Mem0 (failure here doesn't affect the answer)
    #     We send the assistant's previous message (if any) as context, so
    #     short replies like "the beach, always" make sense, plus the user's
    #     message. The new answer is left out so the assistant's own
    #     suggestions aren't mistaken for the user's preferences.

    to_remember = []
    history = st.session_state.messages
    if len(history) >= 3 and history[-3]["role"] == "assistant":
        to_remember.append(history[-3])  # the message the user was replying to
    to_remember.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Saving to memory..."):
            memory.add(to_remember, user_id=user_id)
    except Exception as e:
        st.toast(f"Couldn't save to memory: {e}")