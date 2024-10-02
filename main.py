from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from model import predict_orientation
from io import BytesIO
import uvicorn
import os

app = FastAPI()

# Для шаблонов и статики
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Главная страница с формой для загрузки изображения
@app.get("/")
async def main(request: Request):  # Передаем Request в шаблон
    return templates.TemplateResponse("upload.html", {"request": request})

# Загрузка и обработка изображения
@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image_path = 'input_image.png'
    image.save(image_path)

    # Получаем путь к уже повернутому изображению
    rotated_image_path = predict_orientation(image_path)
    if rotated_image_path == 'Ошибка':
        return templates.TemplateResponse("error.html", {"request": request})
    else:
        rotated_image = Image.open(rotated_image_path)

    # Сохраняем повернутое изображение на диск
        output_path = "static/saved_rotated_image.png"
        rotated_image.save(output_path)

    # Создаем URL для доступа к изображению через веб-сервер
        image_url = f"/{output_path}"

    # Рендерим HTML-шаблон с изображением и путем к файлу
        return templates.TemplateResponse("image.html", {
            "request": request,
            "image_url": image_url,
            "image_path": output_path
        })
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)