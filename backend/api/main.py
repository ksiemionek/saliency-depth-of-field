from fastapi import FastAPI, UploadFile
from fastapi.responses import Response

app = FastAPI(title="Saliency Depth of Field API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(image: UploadFile) -> Response:
    data = await image.read()
    return Response(content=data, media_type=image.content_type)
