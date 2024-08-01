from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Request
import openai
import requests
import deepl
import json
import uvicorn
from pyngrok import ngrok
import threading
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

openai.api_key = OPENAI_API_KEY
DEEPL_API_URL = 'https://api-free.deepl.com/v2/translate'

app = FastAPI()

# CORS 설정 
origins = ["http://localhost:3000", "https://sproupt.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def translate_text(text, target_lang):
    data = {
        'auth_key': DEEPL_API_KEY,
        'text': text,
        'target_lang': target_lang
    }
    response = requests.post(DEEPL_API_URL, data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Translation API error")
    return response.json()['translations'][0]['text']

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal::9rL2MypF",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def handle_user_query(user_query):
    translated_text = translate_text(user_query, target_lang="KO")
    print("user: ", translated_text)

    gpt_response_korean = get_gpt_response(translated_text)
    gpt_response_chinese = translate_text(gpt_response_korean, target_lang="ZH")

    return gpt_response_korean, gpt_response_chinese

@app.post("/api/chat")
async def chat_with_gpt(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # 사용자 질문 전달 및 번역 처리
        gpt_response_korean, gpt_response_chinese = handle_user_query(prompt)

        return {
            "korean_response": gpt_response_korean,
            "chinese_response": gpt_response_chinese
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ngrok 설정 및 FastAPI 실행
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8001)  # 포트를 8001로 변경
print(f"Public URL: {public_url}")

def run():
    uvicorn.run(app, host="0.0.0.0", port=8001)  # 포트를 8001로 변경

# FastAPI 서버를 별도의 스레드에서 실행
thread = threading.Thread(target=run)
thread.start()
