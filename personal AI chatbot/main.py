import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from datetime import datetime , timedelta
import sqlite3
import json
import os
import anthropic
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph ,END
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from sentence_transformers import SentenceTransformer
from transformers import pipeline 

st.title("Personal AI")
input_text = st.text_input("chat with me")

llm = ChatOllama(model_id = "mistral")
output = StrOutputParser()

con = sqlite3.connect("chatbot database")
cr = con.cursor()
cr.execute("""
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_message TEXT,
    bot_response TEXT,
    category TEXT,
    mood TEXT,
    theme TEXT,
    goal TEXT
)
""")

memory_folder = "memory"
os.makedirs(memory_folder, exist_ok=True)

prompt_template = PromptTemplate.from_template("Q: {Question}\nA: ...")

category_dynamic_prompt = PromptTemplate.from_template("""
Message: {message}

Category:
""") 

mood_theme_detact = PromptTemplate.from_template("""
Message: "{message}"

respond in json data:
{{
    "mood": "<mood>",
    "theme": "<theme>",
    "goal": "<goal>"
}}                    
""")

custom_template = PromptTemplate.from_template("""
user:{profile}
                                               
input:
{input_text}
""")

embedding_model = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

cr.execute("SELECT timestamp, user_message, bot_response FROM chat_logs")
rows = cr.fetchall()

vector_store = None
docs = []
for timestamp, user_meg, bot_res in rows:
    text = f"user :{user_meg}\n bot:{bot_res}"
    time_date = {"timestamp": timestamp}
    docs.append(Document(page_content=text, metadata=time_date))

if docs:
    vector_store = Chroma.from_documents(docs, embedding_model, persist_directory="./vector_db")
    vector_store.persist()

def my_profile(memory_folder=memory_folder):
    profil = {}
    for file_name in os.listdir(memory_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(memory_folder, file_name)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    key = file_name.replace(".json", "")
                    profil[key] = data
                except Exception as e:
                    print(f"error:{e}")
    return profil

def ask_insightful_question(state):
    profile = my_profile()
    recent_summary = vector_summarize()
    input_context = f"""
    User Profile: {json.dumps(profile, indent=2)}
    Weekly Summary: {recent_summary}
    Last User Input: {state['input_text']}
    """
    prompt = f"inputs.Context:\n{input_context}\n\nOutput only the question."
    state_with_prompt = {"prompt": prompt}
    response = llm_invoke(state_with_prompt)
    follow_up = response["llm_response"]
    st.markdown(f"Follow-up Question: {follow_up}")
    return {**state, "follow_up_question": follow_up}

def weekly_time(days=7):
    vector_store = Chroma(persist_directory="./vector_db", embedding_function=embedding_model)
    recent_time = datetime.now() - timedelta(days=days)
    all_docs = vector_store.similarity_search("summary", k=100)
    return [doc for doc in all_docs if datetime.fromisoformat(doc.metadata["timestamp"]) >= recent_time]

def vector_summarize():
    weekly_docs = weekly_time()
    if not weekly_docs:
        return "not found chats"
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(weekly_docs)

if st.button("weekly summary"):
    st.info("searching for summary...")
    summary = vector_summarize()
    st.markdown(summary)

def replay_month(days=30):
    vector_store = Chroma(persist_directory="./vector_db", embedding_function=embedding_model)
    current_time = datetime.now() - timedelta(days=days)
    all_docs = vector_store.similarity_search("summary", k=500)
    return [doc for doc in all_docs if datetime.fromisoformat(doc.metadata["timestamp"]) >= current_time]

def vector_replaymonth_summarize():
    replay_docs = replay_month()
    if not replay_docs:
        return "not found chats"
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(replay_docs)

if st.button("Replay my month"):
    st.info("searching for summary...")
    summary = vector_replaymonth_summarize()
    st.markdown(summary)

def prompt_temp(state):
    prompt = prompt_template.format(Question=state["input_text"])
    return {"prompt": prompt, **state}

def llm_invoke(state):
    response = llm([HumanMessage(content=state["prompt"])])
    content = response.content if hasattr(response, "content") else response
    return {"llm_response": content, **state}

def parsed_output(state):
    text = state["llm_response"]
    output_text = output.parse(text)
    return {"output": output_text, **state}

def category_decide(state):
    prompt = prompt_template.format(Question=state["input_text"])
    response = llm_invoke({"prompt": prompt})
    category = response["llm_response"].strip().lower().replace(" ", "_")
    return {"category": category, **state}

def mood_theme(state):
    prompt = mood_theme_detact.format(message=state["input_text"])
    try:
        response = llm_invoke({"prompt": prompt})
        result = json.loads(response["llm_response"])
    except:
        result = {"mood": "unknown", "theme": "unknown", "goal": "unknown"}
    return {"mood": result["mood"], "theme": result["theme"], "goal": result["goal"], **state}

def write_my_blog(state):
    memory_files = os.listdir(memory_folder)
    thought = []
    for file in memory_files:
        with open(os.path.join(memory_folder, file), "r") as f:
            thought.extend(json.load(f))
    content = "\n\n".join([f"{item['message']}\n{item['response']}" for item in thought])
    prompt = f"write blogs and notes :\n\n{content}"
    response = llm(prompt)
    blog_text = response.content if hasattr(response, "content") else str(response)
    with open(os.path.join(memory_folder, "blog_output.txt"), "w") as f:
        f.write(blog_text)
    return {"output_text": blog_text, **state}

if st.button("write my blog and notes"):
    st.info("creating blog and notes...")
    blog_notes = write_my_blog({})
    st.markdown(blog_notes["output_text"])

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def ai_therapist(state):
    emotion_detect = emotion_classifier(state["input_text"])[0]
    emotion = emotion_detect['label']
    prompt = f"emotion : {emotion} .\n\n user:{state['input_text']}"
    response = llm_invoke({"prompt": prompt})
    output_text = response["llm_response"]
    return {"output_text": output_text, **state}

def get_responce_profil(state):
    profile = my_profile()
    prompt = custom_template.format(profile=json.dumps(profile, indent=2), input_text=state["input_text"])
    response = llm(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"input_text": state["input_text"], "output_text": content}

def storage(state):
    file_path = f"{memory_folder}/category.json"
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    data.append({"message": state["input_text"], "response": state["output_text"]})
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    cr.execute("INSERT INTO chat_logs (timestamp, user_message, bot_response, category, mood, theme, goal) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (str(datetime.now()), state["input_text"], state["output_text"], state["category"], state["mood"], state["theme"], state["goal"]))
    con.commit()
    return state

def final(state):
    st.markdown(f"Bot: {state['output']}")
    st.caption(f"Stored in dynamic category: `{state['category']}`| Mood:`{state['mood']}`| Theme:`{state['theme']}`| goal:`{state['goal']}`")

graph = StateGraph(state_schema=dict)
graph.set_entry_point("prompt_temp")
graph.add_node("prompt_temp", prompt_temp)
graph.add_node("llm_invoke", llm_invoke)
graph.add_node("parsed_output", parsed_output)
graph.add_node("category_decide", category_decide)
graph.add_node("mood_theme", mood_theme)
graph.add_node("write_my_blog", write_my_blog)
graph.add_node("ai_therapist", ai_therapist)
graph.add_node("get_responce_profil", get_responce_profil)
graph.add_node("ask_insightful_question", ask_insightful_question)
graph.add_node("storage", storage)
graph.add_node("show_output", final)


graph.add_edge("prompt_temp", "llm_invoke")
graph.add_edge("llm_invoke", "parsed_output")
graph.add_edge("parsed_output", "category_decide")
graph.add_edge("category_decide", "mood_theme")
graph.add_edge("mood_theme", "write_my_blog")
graph.add_edge("write_my_blog", "ai_therapist")
graph.add_edge("ai_therapist", "get_responce_profil")
graph.add_edge("get_responce_profil", "ask_insightful_question")
graph.add_edge("ask_insightful_question", "storage")
graph.add_edge("storage", "show_output")
graph.add_edge("show_output", END)
workflow = graph.compile()

if input_text:
    workflow.invoke({"input_text": input_text})
