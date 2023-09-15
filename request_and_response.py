from pydantic import BaseModel

class Request(BaseModel):
    input: str
    src_lang: str
    tgt_lang: str

class Response(BaseModel):
    output: str
