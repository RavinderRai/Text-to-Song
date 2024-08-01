# Text to Music Generator App

This post is a description of a mini-project on a FastAPI app that let's you generate music from just text alone. It takes lyrics as text input and will generate music based off of that, but it goes beyond that a bit by first letting you generate lyrics by describing the type of music you'd like to listen to.

Credits go to [Anytime AI](https://www.youtube.com/watch?v=SbRC81kZBkE) for the original prject, and this is just a modified/extended version of it. The way it works is:

 - The user will enter a description of the type of music they want to listen to.
 - Then we will use OpenAI to generate lyrics with a custom prompt in the backend.
 - This will then enter into another text box for the user to see the lyrics and they can edit them.
 - Alternatively, the user can skip to this part and add their own lyrics immediately.
 - Either way, then we can generate music using the [suno-ai/bark model](https://replicate.com/suno-ai/bark?input=python).

 The only code to really go over is a short app.py file, so let's break it down now, but here is the GitHub repo: https://github.com/RavinderRai/Text-to-Song.

## FastAPI App

### Getting Started

Before we can get started, make sure you have a .env file with an OpenAI API key ready. You will also then need to get an API key from replicate, which will be free initially (see the suno-ai/bark model link above).

Then, here are all the imports you'll need.


```python
import os
from dotenv import load_dotenv
import replicate
from openai import OpenAI
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
```

First, here is the easy part of the code to get us started. It simply creates the FastAPI app and loads our environment variables. It also get's out html template going for the frontend. We won't cover that here, but it's a standard template which you can take from the GitHub repo and modify for your own project.


```python
app = FastAPI()

load_dotenv()

replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

app.mount("/static", StaticFiles(directory = "static"), name = "static")

templates = Jinja2Templates(directory = "templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

```

### Generating Lyrics

Now we can generate our lyrics. So taking the user prompt, describing the style of music they want lyrics for, we need to prompt OpenAI accordingly. I stuck with Anytime AI's original prompt:

"You are a music lyrics writer and your task is to write lyrics of music under 30 words based on user's prompt. Just return the lyrics and nothing else."

But feel free to modify it. One notable change was the temperature, where I added a slider for the user to decide how creative they want their lyrics to be. This addition can be seen more in the html file, but basically the slider goes from 0-10, with a default value of 5, so we just need to divide it by 10 to get it in the range of 0-1 and we are ready to input it into OpenAI completions function.
            


```python
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
```

And that's it, we get the lyrics and then put it in the next text box where the user can edit them, or at least see them and generate something else if they like.

### Generating Music

Once the lyrics are decided on though, the user will click on the Generate Music button to generate the music (obviously). But we need to do one small step first. Here is the next async function for all of this:


```python
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
```

So basically we need the ♪ character to be surrounding the text for the model to work. Otherwise we can run it using the code you'll find in it's documentation on replicate.

But that's it, then the music will generate (it may take some time though), and you can listen to it. Note that it oftentimes still won't be good, so try different prompts before completely giving up on the model!
