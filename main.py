import panel as pn
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sliders.dash1 import createApp1
from sliders.dash2 import createApp2
from sliders.dash3 import createApp3

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home_page(request: Request):    
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/firstdash")
async def dash1_page(request: Request):
    script = server_document('http://127.0.0.1:5000/dash1')
    return templates.TemplateResponse("dashboard1.html", {"request": request, "script": script})

@app.get("/seconddash")
async def dash2_page(request: Request):
    script = server_document('http://127.0.0.1:5000/dash2')
    return templates.TemplateResponse("dashboard2.html", {"request": request, "script": script})

@app.get("/thirddash")
async def dash3_page(request: Request):
    script = server_document('http://127.0.0.1:5000/dash3')
    return templates.TemplateResponse("dashboard3.html", {"request": request, "script": script})


pn.serve({'/dash1': createApp1, '/dash2': createApp2, '/dash3': createApp3},
         port=5000, allow_websocket_origin=["127.0.0.1:8000"], address="127.0.0.1", show=False)


# uvicorn main:app --reload
# uvicorn main:app --host 127.0.0.1 --port 8000 --reload

