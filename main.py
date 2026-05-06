"""
ВАРИАНТ 1. СЕРВИС КОМПЬЮТЕРНОГО ЗРЕНИЯ

README.MD
"""
import uuid
import os
import logging
import time
import imghdr
import datetime
import aiofiles
from contextlib import asynccontextmanager, suppress
from typing import Any
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from ultralytics import YOLO
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from collections.abc import Sequence

from conf import settings

EXT_ALL = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"]
# ----------------------------------------------------------------------------------------------------------------------
# region Логирование с ротацией


filehandler = TimedRotatingFileHandler(settings.FILE_LOGGING, when="midnight", interval=1, backupCount=14)
filehandler.suffix = "%m-%d"
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(filehandler)
logger.setLevel(settings.LOGGING_LEVEL)

# endregion
# ----------------------------------------------------------------------------------------------------------------------


# region Класс для детекции на изображении
class PredictionService:
    def __init__(self):
        # вот тут зависало, когда 3-4 раза вподряд пытаемся получить модель, как будто по IP блок получал
        self.model = YOLO(settings.APP_YOLO)

    def predict(self, photo: str, conf: float = 0.25) -> dict[str, list[float]] | None:
        result = self.model(source=photo, conf=conf, save=False, verbose=False)[0]
        classes_names = result.names
        classes = result.boxes.cls.cpu().numpy()
        classes_conf = result.boxes.conf.cpu().numpy()

        # итоговый результат
        grouped_objects = {}
        for i, class_id in enumerate(classes):
            class_name = classes_names[int(class_id)]
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(round(float(classes_conf[i]), 2))

        return grouped_objects
        # return {"person": [0.12, 0.11]}

# endregion
# ----------------------------------------------------------------------------------------------------------------------
# region Создаем FastAPI-приложение, определяем маршруты


async def on_startup(app_type: FastAPI):
    """
    Выполняем при старте сервиса
    """
    logger.info("START SERVICE".center(60, "-"))
    logger.info(f"Модель YOLO: {settings.APP_YOLO}")
    app_type.state.model = PredictionService()
    logger.info(f"Модель YOLO: {settings.APP_YOLO} = OK")


async def on_shutdown():
    """
    Выполняем при остановке сервиса
    """
    logger.info('SHUTTING DOWN'.center(60, "-"))


@asynccontextmanager
async def lifespan(app_type: FastAPI):
    """
    Жизненный цикл
    """
    await on_startup(app_type)
    yield
    await on_shutdown()


app = FastAPI(
    title="CV YOLO",
    description="Детекция объектов на изображении",
    version="1.0",
    lifespan=lifespan,
)


# Разрешает принимать запросы к API со всех доменов (или только с localhost).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "http://localhost:8000"
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """
    Обрабатываем каждый HTTP-запрос перед endpoint
    Засекаем время, добавляем id запроса
    :param request:   запрос
    :param call_next: вызывает следующий middleware или сам endpoint
    """
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - started:.6f}"
    response.headers["X-Request-Id"] = request_id
    return response


# проверяем состояние приложения | tags для документации свагера
@app.get("/health", tags=["system"])
async def health(request: Request):
    logger.info(f"GET: health[{getattr(request.state, 'request_id', None)}] | host[{request.client.host}]")
    return {"status": "ok", "model": settings.APP_YOLO, "env": settings.APP_ENV}


def rand_name(ext: str) -> str:
    """
    Получаем рандомное имя файла
    """
    return f"{settings.IMG_TMP}/{str(round(datetime.datetime.now().timestamp(), 3)).replace('.', '_')}.{ext}"


async def check_photo(photo: UploadFile) -> (bytes, str) or None:
    """
    Проверяем валидность изображения. Но даже тут можно это обойти, сформировав подставной файл
    """
    # первый уровень проверки фото по расширению
    suffix = Path(photo.filename or "").suffix.lower()[1:]  # убираем точку
    if suffix not in EXT_ALL:
        raise HTTPException(415, f"Неподдерживаемый формат: {suffix}({photo.filename})")

    # проверяем размер
    size_i = settings.MAX_PHOTO_SIZE * 1024 * 1024
    if photo.size > size_i:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Maximum size allowed is {size_i // (1024 * 1024)}MB."
                                   f"({photo.filename})")

    # злоумышленник может подделать файл, нужно проверить содержимое
    content = await photo.read()
    if (ext_s := imghdr.what("", h=content)) not in EXT_ALL:
        raise HTTPException(415, f"Неподдерживаемый формат: {ext_s}({photo.filename})")

    return content, suffix


# запрос с файлом и величиной conf (порог уверенности)
@app.post("/predict", response_model=dict[str, list[float]] | None, tags=["yolo"])
async def predict(request: Request,
                  conf: float | None = Form(default=0.25, ge=0.01, le=0.99),
                  photo: UploadFile = File(...)):

    logger.info(f"POST: predict[{getattr(request.state, 'request_id', None)}] | host[{request.client.host}] | "
                f"photo[{photo.filename}] | conf[{conf}]")

    content, ext_s = await check_photo(photo)

    # сохранить файл в папку, а потом не забыть его удалить. Вообще, действие лишнее, но для обучения сойдет
    tmp_photo = rand_name(ext_s)
    try:
        # async write
        async with aiofiles.open(tmp_photo, 'wb') as out_file:
            await out_file.write(content)

        # получаем ответ
        # res_d = get_prediction_service(tmp_photo, conf).predict()
        res_d = app.state.model.predict(tmp_photo, conf)
        logger.info(f"POST: predict[{getattr(request.state, 'request_id', None)}] | OK: {res_d}")
        return res_d
    finally:
        with suppress(Exception):
            os.remove(tmp_photo)

# endregion
# ----------------------------------------------------------------------------------------------------------------------
# region Обработка ошибок


def build_error_payload(request: Request, detail: str, error_type: str,
                        errors: Sequence | None = None) -> dict[str, Any]:
    """
    Формирование ответа ошибки
    """
    res: dict[str, Any] = {"detail": detail, "error_type": error_type,
                           "request_id": getattr(request.state, "request_id", None)}
    if errors is not None:
        res["errors"] = errors
    return res


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Ошибки валидации, которые возникают при проверке входных данных
    """
    logger.error(f"Exception:RequestValidationError({getattr(request.state, 'request_id', None)}): "
                 f"Ошибка валидации запроса. {exc.errors()}")
    return JSONResponse(status_code=422,
                        content=build_error_payload(request,
                                                    detail="Ошибка валидации запроса",
                                                    error_type="validation_error", errors=exc.errors()))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Ошибки клиента, некорректная аутентификация, некорректные данные
    """
    logger.error(f"Exception:HTTPException({getattr(request.state, 'request_id', None)}): {exc.detail}")
    return JSONResponse(status_code=exc.status_code,
                        content=build_error_payload(request, detail=str(exc.detail), error_type="http_error"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Остальные ошибки
    """
    logger.error(f"Exception:Exception({getattr(request.state, 'request_id', None)}): {repr(exc)}")
    return JSONResponse(status_code=500,
                        content=build_error_payload(request,
                                                    detail=f"Exception: {repr(exc)}",
                                                    error_type="exception_error"))

# endregion

# end
