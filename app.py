import json
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
from openai import OpenAI
from datetime import datetime


model_pricings = {
    "gpt-4o": {
        "input_tokens": 5.00 / 1_000_000,  # per token
        "output_tokens": 15.00 / 1_000_000,  # per token
    },
    "gpt-4o-mini": {
        "input_tokens": 0.150 / 1_000_000,  # per token
        "output_tokens": 0.600 / 1_000_000,  # per token
    }
}
MODEL = "gpt-4o-mini"
USD_TO_PLN = 4.05
PRICING = model_pricings[MODEL]

# Pole do ręcznego wprowadzenia klucza API
api_key = st.sidebar.text_input("Wpisz swój OpenAI API Key:", type="password")

if not api_key:
    st.error("Musisz podać swój OpenAI API Key, aby korzystać z aplikacji.")
    st.stop()

openai_client = OpenAI(api_key=api_key)

#
# CHATBOT
#
def prepare_conversation_context(messages, max_tokens=4000):
    """Przygotowuje kontekst konwersacji z inteligentnym obcinaniem historii."""
    token_count = 0
    context = []
    
    # Zawsze dodaj ostatnią wiadomość użytkownika
    if messages:
        context.append(messages[-1])
        
    # Dodawaj wcześniejsze wiadomości, dopóki nie przekroczysz limitu tokenów
    for msg in reversed(messages[:-1]):
        estimated_tokens = len(msg["content"].split()) * 1.3  # przybliżona liczba tokenów
        if token_count + estimated_tokens > max_tokens:
            break
        context.insert(0, msg)
        token_count += estimated_tokens
        
    return context

def chatbot_reply(user_prompt, memory):
    # dodaj system message
    messages = [
        {
            "role": "system",
            "content": st.session_state["chatbot_personality"],
        },
    ]
    
    # Użyj inteligentnego zarządzania kontekstem zamiast sztywnej liczby wiadomości
    context = prepare_conversation_context(memory)
    
    # dodaj wszystkie wiadomości z kontekstu
    for message in context:
        messages.append({"role": message["role"], "content": message["content"]})
    
    # dodaj wiadomość użytkownika jeśli nie jest już w kontekście
    if memory and memory[-1]["role"] == "user" and memory[-1]["content"] == user_prompt:
        pass  # już jest w kontekście
    else:
        messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    usage = {}
    if response.usage:
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }

#
# CONVERSATION HISTORY AND DATABASE
#
DEFAULT_PERSONALITY = """
Jesteś ekspertem w Pythonie, NLP oraz w tworzeniu aplikacji opartych na AI. Pomagasz mi w stworzeniu aplikacji o 
runach nordyckich, która będzie miała następujące funkcje:
Losowanie runy dnia – wraz z interpretacją, radami i wartościowymi opisami.
Losowanie układu run – np. krzyż celtycki, układ partnerski.
Model AI – który będzie interpretował układy run na podstawie źródeł oraz intencji wpisanej przez użytkownika.
Dodatkowo chcę oddzielić dane (opisy run) od kodu i zarządzać nimi w osobnym pliku, np. JSON lub module Pythona. 
Kod powinien być czysty, dobrze udokumentowany i łatwy do rozwijania.
Proszę, abyś:
Doradzał mi najlepsze praktyki w zarządzaniu danymi i organizacji kodu.
Pomógł mi napisać funkcje do pobierania, losowania i interpretacji run.
Wspierał mnie w implementacji prostego modelu NLP do interpretacji układów run.
Sugestie i poprawki kodu podawał w czytelnej formie i tłumaczył, dlaczego warto je zastosować.
Jeśli coś można zrobić lepiej, proponuj optymalne rozwiązania. Bądź konkretny, precyzyjny i pomagaj mi rozwijać 
moje umiejętności w kodowaniu. Możesz zadawać pytania, jeśli coś wymaga doprecyzowania.
""".strip()

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"
EXPORTS_PATH = Path("exports")
# db/
# ├── current.json
# ├── conversations/
# │   ├── 1.json
# │   ├── 2.json
# │   └── ...
# ├── exports/
# │   ├── sokrates_conv_1_20250410_123045.json
# │   └── ...

def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]
    st.session_state["tags"] = conversation.get("tags", [])


def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": "Sokrates 1",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
            "tags": []
        }

        # tworzymy nową konwersację
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
            f.write(json.dumps(conversation))

        # która od razu staje się aktualną
        with open(DB_PATH / "current.json", "w") as f:
            f.write(json.dumps({
                "current_conversation_id": conversation_id,
            }))

    else:
        # sprawdzamy, która konwersacja jest aktualna
        with open(DB_PATH / "current.json", "r") as f:
            data = json.loads(f.read())
            conversation_id = data["current_conversation_id"]

        # wczytujemy konwersację
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
            conversation = json.loads(f.read())

    load_conversation_to_state(conversation)


def save_current_conversation_messages():
    conversation_id = st.session_state["id"]
    new_messages = st.session_state["messages"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "messages": new_messages,
        }))


def save_current_conversation_name():
    conversation_id = st.session_state["id"]
    new_conversation_name = st.session_state["new_conversation_name"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "name": new_conversation_name,
        }))


def save_current_conversation_personality():
    conversation_id = st.session_state["id"]
    new_chatbot_personality = st.session_state["new_chatbot_personality"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "chatbot_personality": new_chatbot_personality,
        }))


def update_conversation_tags(conversation_id, tags):
    """Aktualizuje tagi dla danej konwersacji."""
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())
    
    conversation["tags"] = tags
    
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps(conversation))


def create_new_conversation():
    # poszukajmy ID dla naszej kolejnej konwersacji
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))

    # conversation_ids zawiera wszystkie ID konwersacji
    # następna konwersacja będzie miała ID o 1 większe niż największe ID z listy
    conversation_id = max(conversation_ids) + 1 if conversation_ids else 1
    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    conversation = {
        "id": conversation_id,
        "name": f"Sokrates {conversation_id}",
        "chatbot_personality": personality,
        "messages": [],
        "tags": []
    }

    # tworzymy nową konwersację
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps(conversation))

    # która od razu staje się aktualną
    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }))

    load_conversation_to_state(conversation)
    st.rerun()


def switch_conversation(conversation_id):
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }))

    load_conversation_to_state(conversation)
    st.rerun()


def list_conversations():
    conversations = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        with open(p, "r") as f:
            conversation = json.loads(f.read())
            conversations.append({
                "id": conversation["id"],
                "name": conversation["name"],
                "tags": conversation.get("tags", [])
            })

    return conversations


def export_conversation(conversation_id):
    """Eksportuje konwersację do pliku JSON."""
    if not EXPORTS_PATH.exists():
        EXPORTS_PATH.mkdir()
        
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORTS_PATH / f"sokrates_conv_{conversation_id}_{timestamp}.json"
    
    with open(export_path, "w") as f:
        f.write(json.dumps(conversation, indent=2))
        
    return export_path


def import_conversation(file_path):
    """Importuje konwersację z pliku JSON."""
    with open(file_path, "r") as f:
        conversation = json.loads(f.read())
    
    # Znajdź nowy ID dla importowanej konwersacji
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))
    
    new_id = max(conversation_ids) + 1 if conversation_ids else 1
    conversation["id"] = new_id
    conversation["name"] = f"{conversation['name']} (import)"
    
    with open(DB_CONVERSATIONS_PATH / f"{new_id}.json", "w") as f:
        f.write(json.dumps(conversation))
        
    return new_id


def set_theme():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        color: var(--sidebar-text);
    }
    .stApp {
        background-color: var(--main-bg);
        color: var(--main-text);
    }
    .stButton>button {
        background-color: var(--button-bg);
        color: var(--button-text);
    }
    .stTextInput>div>div>input {
        background-color: var(--input-bg);
        color: var(--input-text);
    }
    .stTextArea>div>div>textarea {
        background-color: var(--input-bg);
        color: var(--input-text);
    }
    .stChat {
        background-color: var(--chat-bg);
    }
    .stChatMessage {
        background-color: var(--chat-message-bg);
        color: var(--chat-message-text);
    }
    :root {
        --main-bg: #f0f2f6;
        --main-text: #31333F;
        --sidebar-bg: #ffffff;
        --sidebar-text: #31333F;
        --button-bg: #4169E1;
        --button-text: white;
        --input-bg: #ffffff;
        --input-text: #31333F;
        --chat-bg: #f0f2f6;
        --chat-message-bg: #ffffff;
        --chat-message-text: #31333F;
    }
    [data-theme="dark"] {
        --main-bg: #0E1117;
        --main-text: #FAFAFA;
        --sidebar-bg: #262730;
        --sidebar-text: #FAFAFA;
        --button-bg: #4169E1;
        --button-text: white;
        --input-bg: #262730;
        --input-text: #FAFAFA;
        --chat-bg: #0E1117;
        --chat-message-bg: #262730;
        --chat-message-text: #FAFAFA;
    }
    </style>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const theme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.checked = theme === 'dark';
        }
    });
    
    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }
    </script>
    
    <!-- Dodanie przycisku kopiowania do bloków kodu -->
    <style>
    .copy-button {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 5px 10px;
        background-color: var(--button-bg);
        color: var(--button-text);
        border: none;
        border-radius: 3px;
        cursor: pointer;
        z-index: 999;
    }
    .copy-button:hover {
        opacity: 0.9;
    }
    .code-block {
        position: relative;
    }
    </style>
    
    <script>
    function copyCode(button) {
        const codeBlock = button.parentElement;
        const code = codeBlock.querySelector('code');
        
        // Stwórz tymczasowy element textarea
        const textarea = document.createElement('textarea');
        textarea.value = code.innerText;
        document.body.appendChild(textarea);
        
        // Zaznacz i skopiuj tekst
        textarea.select();
        document.execCommand('copy');
        
        // Usuń element textarea
        document.body.removeChild(textarea);
        
        // Zmień tekst przycisku na potwierdzenie
        const originalText = button.innerText;
        button.innerText = "Skopiowano!";
        setTimeout(() => {
            button.innerText = originalText;
        }, 2000);
    }
    
    // Funkcja dodająca przyciski kopiowania do bloków kodu
    function addCopyButtons() {
        // Znajduje wszystkie bloki kodu
        const codeBlocks = document.querySelectorAll('pre');
        
        codeBlocks.forEach(block => {
            // Dodaj klasę do pozycjonowania
            block.classList.add('code-block');
            
            // Sprawdź, czy przycisk już istnieje
            if (!block.querySelector('.copy-button')) {
                // Utwórz przycisk
                const copyButton = document.createElement('button');
                copyButton.textContent = 'Kopiuj';
                copyButton.className = 'copy-button';
                copyButton.onclick = function() {
                    copyCode(this);
                };
                
                // Dodaj przycisk do bloku kodu
                block.appendChild(copyButton);
            }
        });
    }
    
    // Uruchom funkcję po załadowaniu strony
    document.addEventListener('DOMContentLoaded', addCopyButtons);
    
    // Uruchom funkcję również po aktualizacji DOM (np. po wygenerowaniu nowej odpowiedzi)
    const observer = new MutationObserver(function(mutations) {
        for (const mutation of mutations) {
            if (mutation.addedNodes.length) {
                addCopyButtons();
            }
        }
    });
    
    // Obserwuj zmiany w całym dokumencie
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)


#
# MAIN PROGRAM
#
set_theme()
load_current_conversation()

st.markdown(
    """
    <div style="text-align: center; font-size: 62px; font-weight: bold;">
        🏛️ SOKRATES 🏛️
    </div>
    """,
    unsafe_allow_html=True
)

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("O co chcesz spytać?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = chatbot_reply(prompt, memory=st.session_state["messages"])
        st.markdown(response["content"])

    st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
    save_current_conversation_messages()

with st.sidebar:
    # Dodanie przełącznika motywu
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="margin-right: 10px;">Jasny</span>
        <label class="switch">
            <input type="checkbox" id="theme-toggle" onclick="toggleTheme()">
            <span class="slider round"></span>
        </label>
        <span style="margin-left: 10px;">Ciemny</span>
    </div>
    <style>
    .switch {
      position: relative;
      display: inline-block;
      width: 60px;
      height: 28px;
    }
    .switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #ccc;
      transition: .4s;
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 20px;
      width: 20px;
      left: 4px;
      bottom: 4px;
      background-color: white;
      transition: .4s;
    }
    input:checked + .slider {
      background-color: #2196F3;
    }
    input:checked + .slider:before {
      transform: translateX(32px);
    }
    .slider.round {
      border-radius: 34px;
    }
    .slider.round:before {
      border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Aktualna konwersacja")
    total_cost = 0
    for message in st.session_state.get("messages") or []:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]

    c0, c1 = st.columns(2)
    with c0:
        st.metric("Koszt rozmowy (USD)", f"${total_cost:.4f}")

    with c1:
        st.metric("Koszt rozmowy (PLN)", f"{total_cost * USD_TO_PLN:.4f}")

    st.session_state["name"] = st.text_input(
        "Nazwa konwersacji",
        value=st.session_state["name"],
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )
    
    # System tagowania konwersacji
    tags_input = st.text_input(
        "Tagi (oddzielone przecinkami)",
        value=", ".join(st.session_state.get("tags", [])),
        key="new_tags"
    )

    if st.button("Zapisz tagi"):
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        update_conversation_tags(st.session_state["id"], tags)
        st.session_state["tags"] = tags
        st.success("Tagi zostały zaktualizowane!")
    
    st.session_state["chatbot_personality"] = st.text_area(
        "Osobowość chatbota",
        max_chars=2000,
        height=200,
        value=st.session_state["chatbot_personality"],
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )

    st.subheader("Eksport/Import konwersacji")
    if st.button("Eksportuj konwersację"):
        export_path = export_conversation(st.session_state["id"])
        st.success(f"Konwersacja wyeksportowana do: {export_path}")
    
    uploaded_file = st.file_uploader("Wybierz plik konwersacji do importu", type=["json"])
    if uploaded_file is not None:
        # Zapisz tymczasowo plik
        temp_path = Path("temp_import.json")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            new_id = import_conversation(temp_path)
            st.success(f"Konwersacja zaimportowana z ID: {new_id}")
            
            if st.button("Przełącz na zaimportowaną konwersację"):
                switch_conversation(new_id)
        except Exception as e:
            st.error(f"Błąd podczas importu: {e}")
        finally:
            # Usuń plik tymczasowy
            if temp_path.exists():
                temp_path.unlink()

    st.subheader("Konwersacje")
    if st.button("Nowa konwersacja"):
        create_new_conversation()

    # pokazujemy do 15 konwersacji
    conversations = list_conversations()
    sorted_conversations = sorted(conversations, key=lambda x: x["id"], reverse=True)
    for conversation in sorted_conversations[:15]:
        col1, col2, col3 = st.columns([7, 2, 2])
        
        with col1:
            conv_name = conversation["name"]
            if conversation["tags"]:
                tags_display = ", ".join([f"#{tag}" for tag in conversation["tags"]])
                st.write(f"{conv_name} - {tags_display}")
            else:
                st.write(conv_name)

        with col2:
            if st.button("Załaduj", key=f"load_{conversation['id']}", 
                       disabled=conversation["id"] == st.session_state["id"]):
                switch_conversation(conversation["id"])
                
        with col3:
            if conversation["id"] != st.session_state["id"]:
                if st.button("Usuń", key=f"delete_{conversation['id']}"):
                    # Tutaj można dodać kod usuwania konwersacji
                    pass
