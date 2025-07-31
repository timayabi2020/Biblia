import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List
import ast
import os
import json

# 🔐 Firebase setup (ensure serviceAccountKey.json is in the same folder)
# cred = credentials.Certificate("serviceAccountKey.json")
# firebase_admin.initialize_app(cred)
cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
initialize_app(cred)
db = firestore.client()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # or ["*"] to allow all origins (use with caution!)
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],              # Authorization, Content-Type, etc.
)

# 📦 Request model
class StudyInput(BaseModel):
    user_id: str
    notes: str

# 🧠 Flashcard model
class Flashcard(BaseModel):
    question: str
    answer: str

# 📦 Response model
class StudySessionResponse(BaseModel):
    summary: str
    flashcards: List[Flashcard]

# 🌐 Create LLM instance
def get_llm():
    return ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

# 🔍 Validate that flashcards returned are in proper format
def validate_flashcards(raw_output: str):
    try:
        parsed = ast.literal_eval(raw_output)
        if isinstance(parsed, list) and all(
            isinstance(item, dict) and "question" in item and "answer" in item
            for item in parsed
        ):
            return parsed
    except Exception:
        pass
    return [{"question": "Could not parse flashcards", "answer": raw_output}]

# 🔄 Analyze notes endpoint
@app.post("/analyze", response_model=StudySessionResponse)
async def analyze_notes(input: StudyInput, authorization: str = Header(None)):
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Missing or invalid auth token")

    llm = get_llm()

    # Step 1: Summarize
    summary_prompt = PromptTemplate.from_template(
        "Summarize these Bible study notes:\n{notes}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run(notes=input.notes)

    # Step 2: Generate Flashcards
    flashcard_prompt = PromptTemplate.from_template(
        "From this Bible summary, create 3 question and answer flashcards in JSON format:\n"
        "{summary}\n"
        "Return as: [{{'question': '...', 'answer': '...'}}, ...]"
    )
    flashcard_chain = LLMChain(llm=llm, prompt=flashcard_prompt)
    flashcards_text = flashcard_chain.run(summary=summary)
    flashcards = validate_flashcards(flashcards_text)

    # Step 3: Save to Firestore
    db.collection("bible_study_sessions").add({
        "user_id": input.user_id,
        "notes": input.notes,
        "summary": summary,
        "flashcards": flashcards
    })

    return {"summary": summary, "flashcards": flashcards}

# 📥 Review endpoint for user's past sessions
@app.get("/review", response_model=List[StudySessionResponse])
async def get_user_sessions(user_id: str, authorization: str = Header(None)):
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Missing or invalid auth token")

    docs = db.collection("bible_study_sessions").where("user_id", "==", user_id).stream()
    sessions = []
    for doc in docs:
        data = doc.to_dict()
        sessions.append({
            "summary": data.get("summary", ""),
            "flashcards": data.get("flashcards", [])
        })
    return sessions