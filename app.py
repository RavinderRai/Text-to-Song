from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from openai import OpenAI
import os
import replicate
from dotenv import load_dotenv

"""useful links:
https://www.youtube.com/watch?v=SbRC81kZBkE
https://replicate.com/suno-ai/bark?input=python
"""

app = FastAPI()

load_dotenv()

replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

app.mount("/static", StaticFiles(directory = "static"), name = "static")

templates = Jinja2Templates(directory = "templates")

def get_openai_lyrics(prompt, temperature):
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
        temperature=temperature,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content
    cleaned_output = output.replace("\n", " ")
    return cleaned_output

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-lyrics")
async def generate_lyrics(prompt: str = Form(...), temperature: int = Form(...)):
    temperature = temperature / 10

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
        temperature=temperature,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    lyrics = response.choices[0].message.content
    lyrics = lyrics.replace("\n", " ")

    return {"lyrics": lyrics}


@app.post("/generate-music")
async def generate_music(generated_lyrics: str = Form(...)):
    formatted_lyrics = f"♪ {generated_lyrics} ♪" # this is a mandatory thing to get the vocals
    
    output = replicate.run(
    "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
    input={
        "prompt": formatted_lyrics,
        "text_temp": 0.7,
        "output_full": False,
        "waveform_temp": 0.7
        }
    )
    print(output)
    music_url = output['audio_out']
    music_path_or_url = music_url
    
    return JSONResponse(content={"url": music_path_or_url})