import streamlit as st
import pandas as pd
import os
from datetime import datetime
import sqlite3
from sqlite3 import Error
import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Constants
CLIENT_IMAGES_FOLDER = "client_images"
DATABASE_FILE = "client_management.db"

# API URLs and Keys
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Ensure folders exist
def ensure_folders():
    os.makedirs(CLIENT_IMAGES_FOLDER, exist_ok=True)

# Database functions
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        return conn
    except Error as e:
        st.error(f"Database error: {e}")
    return conn

def create_tables(conn):
    try:
        cursor = conn.cursor()
        # Create clients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        ''')
        # Create client_forms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS client_forms (
                id INTEGER PRIMARY KEY,
                client_id INTEGER,
                title TEXT,
                date TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (id)
            )
        ''')
        # Create client_info table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS client_info (
                id INTEGER PRIMARY KEY,
                client_id INTEGER,
                form_id INTEGER,
                question TEXT,
                answer TEXT,
                date TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (id),
                FOREIGN KEY (form_id) REFERENCES client_forms (id)
            )
        ''')
        # Create conversation_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id INTEGER PRIMARY KEY,
                client_id INTEGER,
                form_id INTEGER,
                title TEXT,
                date TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (id),
                FOREIGN KEY (form_id) REFERENCES client_forms (id)
            )
        ''')
        # Create conversation_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                date TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions (id)
            )
        ''')
        conn.commit()
    except Error as e:
        st.error(f"Error creating tables: {e}")

def add_column_if_not_exists(conn, table_name, column_name, column_type):
    try:
        cur = conn.cursor()
        # Check if the column exists
        cur.execute(f"PRAGMA table_info({table_name});")
        columns = [info[1] for info in cur.fetchall()]
        if column_name not in columns:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
            conn.commit()
    except Error as e:
        st.error(f"Error adding '{column_name}' column to '{table_name}': {e}")

def insert_or_get_client(conn, client_name):
    try:
        sql = '''
        INSERT OR IGNORE INTO clients (name)
        VALUES (?)
        '''
        cur = conn.cursor()
        cur.execute(sql, (client_name,))
        conn.commit()

        cur.execute("SELECT id FROM clients WHERE name = ?", (client_name,))
        return cur.fetchone()[0]
    except Error as e:
        st.error(f"Error inserting/getting client: {e}")
        return None

def create_client_form(conn, client_id, title):
    try:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = '''
        INSERT INTO client_forms (client_id, title, date)
        VALUES (?, ?, ?)
        '''
        cur = conn.cursor()
        cur.execute(sql, (client_id, title, date))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        st.error(f"Error creating client form: {e}")
        return None

def get_client_forms(conn, client_id):
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, date
            FROM client_forms
            WHERE client_id = ?
            ORDER BY date DESC
        ''', (client_id,))
        rows = cur.fetchall()
        return [{'id': row[0], 'title': row[1], 'date': row[2]} for row in rows] if rows else []
    except Error as e:
        st.error(f"Error fetching client forms: {e}")
        return []

def insert_client_info(conn, client_id, form_id, question, answer):
    try:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = '''
        INSERT INTO client_info (client_id, form_id, question, answer, date)
        VALUES (?, ?, ?, ?, ?)
        '''
        cur = conn.cursor()
        cur.execute(sql, (client_id, form_id, question, answer, date))
        conn.commit()
    except Error as e:
        st.error(f"Error inserting client info: {e}")

def get_client_info(conn, client_id, form_id):
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT question, answer, date
            FROM client_info
            WHERE client_id = ? AND form_id = ?
            ORDER BY date DESC
        ''', (client_id, form_id))
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['question', 'answer', 'date'])
            return df
        else:
            return pd.DataFrame(columns=['question', 'answer', 'date'])
    except Error as e:
        st.error(f"Error fetching client info: {e}")
        return pd.DataFrame(columns=['question', 'answer', 'date'])

def get_client_list(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM clients")
        rows = cur.fetchall()
        return [{'id': row[0], 'name': row[1]} for row in rows] if rows else []
    except Error as e:
        st.error(f"Error fetching client list: {e}")
        return []

def get_conversation_sessions(conn, client_id, form_id):
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, date
            FROM conversation_sessions
            WHERE client_id = ? AND form_id = ?
            ORDER BY date DESC
        ''', (client_id, form_id))
        rows = cur.fetchall()
        return [{'id': row[0], 'title': row[1], 'date': row[2]} for row in rows] if rows else []
    except Error as e:
        st.error(f"Error fetching conversation sessions: {e}")
        return []

def create_conversation_session(conn, client_id, form_id, title):
    try:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = '''
        INSERT INTO conversation_sessions (client_id, form_id, title, date)
        VALUES (?, ?, ?, ?)
        '''
        cur = conn.cursor()
        cur.execute(sql, (client_id, form_id, title, date))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        st.error(f"Error creating conversation session: {e}")
        return None

def get_conversation_history(conn, session_id):
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT role, content, date
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY date ASC
        ''', (session_id,))
        rows = cur.fetchall()
        if rows:
            return [{'role': row[0], 'content': row[1], 'date': row[2]} for row in rows]
        else:
            return []
    except Error as e:
        st.error(f"Error fetching conversation history: {e}")
        return []

def save_conversation_message(conn, session_id, role, content):
    try:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = '''
        INSERT INTO conversation_history (session_id, role, content, date)
        VALUES (?, ?, ?, ?)
        '''
        cur = conn.cursor()
        cur.execute(sql, (session_id, role, content, date))
        conn.commit()
    except Error as e:
        st.error(f"Error saving conversation message: {e}")

# Client image handling functions
def save_client_image(client_name, uploaded_image):
    ensure_folders()
    try:
        img = Image.open(uploaded_image)
        img = img.convert('RGB')
        filename = f"{client_name.replace(' ', '_')}.jpg"
        file_path = os.path.join(CLIENT_IMAGES_FOLDER, filename)
        img.save(file_path)
        return filename
    except Exception as e:
        st.error(f"Error saving client image: {e}")
        return None

def get_client_image_path(client_name):
    filename = f"{client_name.replace(' ', '_')}.jpg"
    file_path = os.path.join(CLIENT_IMAGES_FOLDER, filename)
    return file_path if os.path.exists(file_path) else None

# Functions for Scenario Simulation Builder
def get_latest_news(query):
    if not SERPER_API_KEY:
        st.error("Serper API key is not set.")
        return ""
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "q": query if query else "latest news"
    }
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=data)
        response.raise_for_status()
        search_results = response.json()
        # Extract relevant news articles
        news_articles = ""
        if 'news' in search_results:
            for article in search_results['news'][:5]:
                news_articles += f"- {article['title']}: {article['snippet']}\nURL: {article['link']}\n\n"
        else:
            news_articles = "No news articles found."
        return news_articles
    except requests.RequestException as e:
        st.error(f"Serper API error: {e}")
        return "Error: Unable to fetch latest news."

def call_llm_api(prompt, model_choice):
    if model_choice == "Ollama":
        # Ollama API call
        try:
            response = requests.post(OLLAMA_API_URL, json={"model": "llama2", "prompt": prompt})
            response.raise_for_status()
            return response.json().get("response", "No response from Ollama.")
        except requests.RequestException as e:
            st.error(f"Ollama API error: {e}")
            return "Error: Unable to get response from Ollama."
    elif model_choice == "Groq":
        # Groq API call
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": "llama-2-70b-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            st.error(f"Groq API error: {e}")
            return f"Error: Unable to get response from Groq."
    elif model_choice == "GPT-4o-mini":
        # OpenAI API call
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            st.error(f"OpenAI API error: {e}")
            return f"Error: Unable to get response from OpenAI."
    elif model_choice == "Claude 3.5 Sonnet":
        # Anthropic API call
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "Content-Type": "application/json"
        }
        prompt_formatted = f"\n\nHuman: {prompt}\n\nAssistant:"
        data = {
            "prompt": prompt_formatted,
            "model": "claude-2",
            "max_tokens_to_sample": 1000,
            "temperature": 0.7,
            "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"]
        }
        try:
            response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["completion"]
        except requests.RequestException as e:
            st.error(f"Anthropic API error: {e}")
            return f"Error: Unable to get response from Anthropic."
    else:
        return "Error: Invalid model choice."

def generate_scenario(prompt, model_choice):
    return call_llm_api(prompt, model_choice)

def generate_solution(prompt, model_choice):
    return call_llm_api(prompt, model_choice)

def generate_group_activity(prompt, model_choice):
    return call_llm_api(prompt, model_choice)

# Chatbot function
def chatbot_query(conn, query, conversation_history, model_choice, client_id, form_id):
    client_info = get_client_info(conn, client_id, form_id)
    if client_info.empty:
        st.warning("No information available for this form.")
        return "No data available to analyze."

    # Construct system message with client information
    system_message = f"""You are an AI assistant helping with corporate training needs analysis.
The following is information about the client:

"""
    for index, row in client_info.iterrows():
        system_message += f"**{row['question']}**\n{row['answer']}\n\n"

    system_message += """
Based on the above information, provide insightful analysis and recommendations for the client's training program."""

    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        *[{"role": msg["role"], "content": msg["content"]} for msg in conversation_history],
        {"role": "user", "content": query}
    ]

    # Use the call_llm_api function
    if model_choice in ["Ollama", "Groq", "GPT-4o-mini", "Claude 3.5 Sonnet"]:
        if model_choice == "Ollama":
            # For Ollama, combine messages into a single prompt
            prompt = system_message + "\nConversation history:\n"
            for message in conversation_history:
                prompt += f"{'Human' if message['role'] == 'user' else 'AI'}: {message['content']}\n"
            prompt += f"Human: {query}\nAI:"
            return call_llm_api(prompt, model_choice)
        else:
            # For other models, use messages
            prompt = query  # For simplicity, we'll use query as the prompt
            return call_llm_api(prompt, model_choice)
    else:
        return "Error: Invalid model choice."

# Main function
def main():
    st.set_page_config(page_title="Client Management System", layout="wide")

    if 'selected_client' not in st.session_state:
        st.session_state.selected_client = None

    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None

    if 'selected_form' not in st.session_state:
        st.session_state.selected_form = None

    conn = create_connection()
    if conn is not None:
        create_tables(conn)
        # Ensure 'form_id' column exists in 'client_info' and 'conversation_sessions' tables
        add_column_if_not_exists(conn, 'client_info', 'form_id', 'INTEGER')
        add_column_if_not_exists(conn, 'conversation_sessions', 'form_id', 'INTEGER')
    else:
        st.error("Error! Cannot create the database connection.")
        return

    st.title("Client Learning Management System")
    st.subheader("for People Power - Mauritius")

    # Sidebar for selecting or adding a client
    with st.sidebar:
        st.header("Clients")

        client_list = get_client_list(conn)

        # Display clients with images and names
        for client in client_list:
            client_id = client['id']
            client_name = client['name']
            image_path = get_client_image_path(client_name)
            if image_path:
                img = Image.open(image_path)
                st.image(img, width=100)
            else:
                st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)

            if st.button(client_name, key=f"client_{client_id}"):
                st.session_state.selected_client = client
                st.session_state.selected_form = None  # Reset selected form
                st.session_state.selected_session = None  # Reset selected session

        st.markdown("---")
        if st.button("Add New Client"):
            st.session_state.show_add_client = True
        else:
            st.session_state.show_add_client = False

        if 'show_add_client' in st.session_state and st.session_state.show_add_client:
            new_client_name = st.text_input("Enter Client Name")
            uploaded_image = st.file_uploader("Upload client image", type=["jpg", "png", "jpeg"], key="new_image_uploader")
            if st.button("Save Client"):
                if new_client_name:
                    client_id = insert_or_get_client(conn, new_client_name)
                    if client_id:
                        if uploaded_image:
                            save_client_image(new_client_name, uploaded_image)
                        st.success(f"Client '{new_client_name}' added successfully!")
                        st.session_state.selected_client = {'id': client_id, 'name': new_client_name}
                        st.session_state.selected_form = None
                        st.session_state.selected_session = None
                        st.rerun()
                else:
                    st.warning("Please enter a client name.")

        # Model selection
        st.subheader("Select AI Model")
        model_choice = st.selectbox("AI Model", ["Ollama", "Groq", "GPT-4o-mini", "Claude 3.5 Sonnet"])

        if st.session_state.selected_client:
            st.markdown("---")
            st.subheader("Client Forms")
            client_id = st.session_state.selected_client['id']
            client_forms = get_client_forms(conn, client_id)
            for form in client_forms:
                form_title = form['title'] or f"Form on {form['date']}"
                if st.button(form_title, key=f"form_{form['id']}"):
                    st.session_state.selected_form = form
                    st.session_state.selected_session = None  # Reset selected session
            if st.button("Create New Form"):
                st.session_state.selected_form = {'id': None}

            if st.session_state.selected_form and st.session_state.selected_form.get('id'):
                st.markdown("---")
                st.subheader("Conversations")
                form_id = st.session_state.selected_form['id']
                sessions = get_conversation_sessions(conn, client_id, form_id)
                for session in sessions:
                    session_title = session['title'] or f"Conversation on {session['date']}"
                    if st.button(session_title, key=f"session_{session['id']}"):
                        st.session_state.selected_session = session
                if st.button("Start New Conversation"):
                    st.session_state.selected_session = {'id': None}

    # Create tabs
    tabs = st.tabs(["Main Page", "Scenario Simulation Builder", "Users Guide"])

    with tabs[0]:
        if st.session_state.selected_client:
            client = st.session_state.selected_client
            client_name = client['name']
            client_id = client['id']
            st.header(f"Client: {client_name}")

            # Client Image
            image_path = get_client_image_path(client_name)
            if image_path:
                st.image(image_path, width=300, caption=f"{client_name}'s image")
            else:
                st.image("https://via.placeholder.com/300x400?text=No+Image", width=300, caption="No image available")

            if st.session_state.selected_form:
                if st.session_state.selected_form['id'] is None:
                    # Create new form
                    form_title = st.text_input("Form Title", value=f"Form on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    if st.button("Create Form"):
                        form_id = create_client_form(conn, client_id, form_title)
                        st.session_state.selected_form = {'id': form_id, 'title': form_title}
                        st.rerun()
                else:
                    form = st.session_state.selected_form
                    form_id = form['id']
                    st.subheader(f"Form: {form['title']}")

                    # Collect data from the user about the client's corporate training needs
                    st.subheader("Business Needs and Goals")
                    business_questions = [
                        "What are the key business challenges or opportunities your organisation is currently facing?",
                        "What are the desired outcomes of this training program? (e.g., improved productivity, enhanced decision-making, increased innovation, stronger teamwork)",
                        "What specific skills or knowledge gaps do you want to address through this training?",
                        "What is your organisation's industry and what are the key competitive dynamics you are facing?",
                        "Are there any specific industry benchmarks or best practices you would like the training to incorporate?"
                    ]

                    st.subheader("Learner Profile and Needs")
                    learner_questions = [
                        "Who is the target audience for this training program? (e.g., job roles, seniority levels, departments)",
                        "What is the approximate number of participants you plan to include in the training?",
                        "What are the participants' current levels of knowledge and skills related to the training topic?",
                        "What are the learning styles and preferences of your target audience? (e.g., visual, auditory, kinesthetic, hands-on activities, group discussions)",
                        "What are the participants' expectations and motivations for attending this training?"
                    ]

                    st.subheader("Training Preferences")
                    training_questions = [
                        "What level of interactivity and participant engagement do you envision for this training program?",
                        "What types of activities or learning methods would you like to see incorporated to promote active participation and knowledge retention? (e.g., business simulations, case studies, group discussions, role-playing)",
                        "How do you envision participants applying the knowledge and skills they gain from this training in their day-to-day work?"
                    ]

                    all_questions = business_questions + learner_questions + training_questions

                    # Retrieve existing answers
                    existing_info = get_client_info(conn, client_id, form_id)
                    if not existing_info.empty:
                        existing_answers = dict(zip(existing_info['question'], existing_info['answer']))
                    else:
                        existing_answers = {}

                    with st.form("client_info_form"):
                        responses = {}
                        for question in all_questions:
                            responses[question] = st.text_area(question, value=existing_answers.get(question, ""))

                        if st.form_submit_button("Submit"):
                            for question, answer in responses.items():
                                if answer:
                                    # Delete existing answer for the question in this form
                                    cur = conn.cursor()
                                    cur.execute('''
                                        DELETE FROM client_info
                                        WHERE client_id = ? AND form_id = ? AND question = ?
                                    ''', (client_id, form_id, question))
                                    conn.commit()
                                    insert_client_info(conn, client_id, form_id, question, answer)
                            st.success("Client information saved successfully!")
                            st.rerun()

                    # Display collected information
                    st.subheader("Collected Information")
                    client_info = get_client_info(conn, client_id, form_id)
                    if not client_info.empty:
                        # Optionally rename columns for display
                        display_info = client_info.rename(columns={'question': 'Question', 'answer': 'Answer', 'date': 'Date'})
                        st.table(display_info)
                    else:
                        st.write("No information collected yet.")

                    if st.session_state.selected_session:
                        session = st.session_state.selected_session
                        if session['id'] is None:
                            # Start a new conversation
                            session_title = st.text_input("Conversation Title", value=f"Conversation on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            if st.button("Create Conversation"):
                                session_id = create_conversation_session(conn, client_id, form_id, session_title)
                                st.session_state.selected_session = {'id': session_id, 'title': session_title}
                                st.rerun()
                        else:
                            session_id = session['id']
                            # Load conversation history for the selected session
                            conversation_history = get_conversation_history(conn, session_id)

                            # Display conversation history
                            st.subheader(f"Conversation: {session['title']}")
                            for message in conversation_history:
                                with st.chat_message(message["role"]):
                                    st.write(message["content"])

                            chat_query = st.chat_input("Ask about the client's needs or provide recommendations:")
                            if chat_query:
                                # Append user's message to the conversation history
                                conversation_history.append({"role": "user", "content": chat_query})
                                save_conversation_message(conn, session_id, "user", chat_query)
                                with st.chat_message("user"):
                                    st.write(chat_query)

                                with st.chat_message("assistant"):
                                    with st.spinner("Thinking..."):
                                        response = chatbot_query(conn, chat_query, conversation_history, model_choice, client_id, form_id)
                                    st.write(response)

                                # Append assistant's response to the conversation history
                                conversation_history.append({"role": "assistant", "content": response})
                                save_conversation_message(conn, session_id, "assistant", response)
            else:
                st.write("Please select a form from the sidebar or create a new one.")

    with tabs[1]:
        st.header("Scenario Simulation Builder")

        if st.session_state.selected_client and st.session_state.selected_form:
            client = st.session_state.selected_client
            client_name = client['name']
            client_id = client['id']
            form = st.session_state.selected_form
            form_id = form['id']

            # Retrieve training preferences from client_info
            client_info = get_client_info(conn, client_id, form_id)
            if client_info.empty:
                st.warning("No information available for this form.")
            else:
                # Extract training preferences
                training_preferences_questions = [
                    "What level of interactivity and participant engagement do you envision for this training program?",
                    "What types of activities or learning methods would you like to see incorporated to promote active participation and knowledge retention? (e.g., business simulations, case studies, group discussions, role-playing)",
                    "How do you envision participants applying the knowledge and skills they gain from this training in their day-to-day work?"
                ]

                training_preferences = ""
                for question in training_preferences_questions:
                    answers = client_info[client_info['question'] == question]['answer']
                    if not answers.empty:
                        training_preferences += f"**{question}**\n{answers.iloc[0]}\n\n"

                if not training_preferences:
                    st.warning("No training preferences available to generate simulation.")
                else:
                    # Require the user to enter a topic
                    st.subheader("Scenario Simulation")

                    # Make topic input required
                    query = st.text_input("Enter a topic for the scenario simulation:")

                    if not query:
                        st.warning("Please enter a topic to generate the scenario.")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Generate Scenario Simulation"):
                                with st.spinner("Generating scenario simulation..."):
                                    # Get news articles related to the topic
                                    news_articles = get_latest_news(query)

                                    # Prepare the prompt for the LLM, including the topic
                                    prompt = f"""You are an AI assistant tasked with creating a real-world scenario simulation for corporate training focusing on the topic: "{query}".
The simulation should be based on the following training preferences:

{training_preferences}

Incorporate the latest news and current affairs related to "{query}" into the simulation to make it relevant.

Here are some recent news articles:

{news_articles}

Please create a realistic and engaging scenario that challenges students to navigate and develop the best solution."""
                                    # Generate the scenario using the selected LLM
                                    scenario = generate_scenario(prompt, model_choice)

                                    if scenario:
                                        st.session_state['scenario'] = scenario
                                        st.session_state['solution'] = None  # Clear any previous solution
                                    else:
                                        st.error("Failed to generate scenario simulation.")
                                        st.session_state['scenario'] = None
                        with col2:
                            if st.button("Create Group Activity"):
                                with st.spinner("Generating group activity..."):
                                    # Prepare the prompt for the group activity
                                    prompt = f"""You are an AI assistant tasked with creating a group activity for corporate training focusing on the topic: "{query}".
The activity should be interactive and engaging, and based on the following training preferences:

{training_preferences}

Please create a detailed group activity that facilitates learning and collaboration among participants."""
                                    # Generate the group activity using the selected LLM
                                    group_activity = generate_group_activity(prompt, model_choice)

                                    if group_activity:
                                        st.session_state['group_activity'] = group_activity
                                    else:
                                        st.error("Failed to generate group activity.")
                                        st.session_state['group_activity'] = None

                    if 'scenario' in st.session_state and st.session_state['scenario']:
                        st.subheader("Generated Scenario Simulation")
                        st.write(st.session_state['scenario'])

                        if st.button("Get Solution"):
                            with st.spinner("Generating solution..."):
                                # Generate the solution, ensuring the topic is considered
                                solution_prompt = f"""Based on the following scenario focusing on the topic "{query}":

{st.session_state['scenario']}

Provide the best possible solution to navigate and resolve the challenges presented."""
                                solution = generate_solution(solution_prompt, model_choice)

                                if solution:
                                    st.session_state['solution'] = solution
                                else:
                                    st.error("Failed to generate solution.")
                                    st.session_state['solution'] = None

                    if 'solution' in st.session_state and st.session_state['solution']:
                        st.subheader("Solution")
                        st.write(st.session_state['solution'])

                    if 'group_activity' in st.session_state and st.session_state['group_activity']:
                        st.subheader("Generated Group Activity")
                        st.write(st.session_state['group_activity'])
        else:
            st.warning("Please select a client and a form on the main page.")

    with tabs[2]:
        st.header("Users Guide")
        # Read and display the content of guide.html
        try:
            with open("guide.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            # Use st.components.v1.html to render full HTML content
            st.components.v1.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Error loading guide.html: {e}")

    conn.close()

if __name__ == "__main__":
    main()
