from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import OpenAI
import torch
import os
from dotenv import load_dotenv

"""
https://www.youtube.com/watch?v=SbRC81kZBkE
https://replicate.com/suno-ai/bark?input=python
"""

app = FastAPI()

replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

app.mount("/static", StaticFiles(directory = "static"), name = "static")

templates = Jinja2Templates(directory = "templates")

def generate_lyrics(prompt):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a music lyrics writer and your task is to write lyrics of \
                music under 30 words based on user's prompt. Just return the lyrics and nothing else.",
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪" # this is a mandatory thing to get the vocals
    return formatted_lyrics

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("\generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form( ...)):
    lyrics = generate_lyrics(prompt)
    prompt_with_lyrics = lyrics
    print(prompt_with_lyrics)

    output = replicate.run(
    "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
    input={
        "prompt": "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.",
        "text_temp": 0.7,
        "output_full": False,
        "waveform_temp": 0.7,
        "history_prompt": "announcer"
        }
    )
    print(output)