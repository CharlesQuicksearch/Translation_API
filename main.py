from fastapi import FastAPI, HTTPException

from translator import translate
from request_and_response import Request, Response

app = FastAPI()

@app.get("/")
async def home():
    return {"Translation API. Call with strings: 'input','src_language','tgt_language'"}

@app.post("/translate_string/", response_model=Response)
async def translate_string(request_data: Request):
    try:
        translated_string = await translate(
            request_data.input,
            request_data.src_lang,
            request_data.tgt_lang)
        return Response(output=translated_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
