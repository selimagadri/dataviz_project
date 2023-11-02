
import panel as pn
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sliders.pn_app import createApp, createApp1  # Import your Panel app from sliders.pn_app
from sliders.dash1 import createApp2

app = FastAPI()
# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def page(request: Request):    
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/firstpage")
async def bkapp_page(request: Request):
    script = server_document('http://127.0.0.1:5000/app')
    return templates.TemplateResponse("base.html", {"request": request, "script": script})

@app.get("/otherpage")
async def other_page(request: Request):
    script = server_document('http://127.0.0.1:5000/other')
    return templates.TemplateResponse("base.html", {"request": request, "script": script})


pn.serve({'/app': createApp2, '/other': createApp1},
        port=5000, allow_websocket_origin=["127.0.0.1:8000"],
         address="127.0.0.1", show=False)

#uvicorn main:app --reload

if __name__ == "__main__":
    import uvicorn

    # Serve both Panel apps
    pn.serve({'/app': createApp2, '/other': createApp1},
        port=5000, allow_websocket_origin=["127.0.0.1:8000"],
         address="127.0.0.1", show=False)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
