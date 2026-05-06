"""
ВАРИАНТ 1. СЕРВИС КОМПЬЮТЕРНОГО ЗРЕНИЯ

Задача: разработать REST-сервис для решения практической задачи компьютерного зрения - детекция объектов на изображении

Задание 1. Постановка задачи и данные
    Ищем на изображении объекты, указывая величину от 0 до 1 - уверенность в правильности детекции.
    Объекты (80 штук): человек, велосипед, машина, мотоцикл, самолет, автобус, поезд, грузовик, лодка, светофор,
        пожарный гидрант, знак остановки, парковочный счетчик, скамейка, птица, кошка, собака, лошадь, овца, корова,
        слон, медведь, зебра, жираф, рюкзак, зонт, сумочка, галстук, чемодан, фрисби, лыжи, сноуборд, спортивный мяч,
        воздушный змей, бейсбольная бита, бейсбольная перчатка, скейтборд, серфборд, теннисная ракетка, бутылка,
        бокал для вина, чашка, вилка, нож, ложка, миска, банан, яблоко, сэндвич, апельсин, брокколи, морковь, хот-дог,
        пицца, пончик, торт, стул, диван, комнатное растение, кровать, обеденный стол, туалет, телевизор, ноутбук, мышь,
        пульт дистанционного управления, клавиатура, мобильный телефон, микроволновая печь, духовка, тостер, раковина,
        холодильник, книга, часы, ваза, ножницы, плюшевый мишка, фен, зубная щетка
    Изображение подается в формате .jpg, .jpeg, .png, .webp, .bmp, .tif, .tiff (todo: на счет .gif надо проверить)
        Размер изображения не больше 5МБ. Можно это сделать на стороне клиента (javascript:file.size)
        или на стороне сервера (Content-Length в заголовке) до полной загрузки файла. Но эти проверки злоумышленник
        может обойти.
    Датасет будет состоять из изображений разных форматов и размеров. Так же в датасет будут включены файлы, которые
        должны вызвать исключения в процессе работы.

Задание 2. Подготовка и использование модели CV
    В качестве модели возьмем предобученную YOLO v8 small (yolov8s.pt).
    Этой модели будем передавать изображение как путь к файлу, тогда конвертация в RGB будет автоматической.
    Плюсом будет, что не нужно на вход модели передавать изображение определенного размера. Модель сама всё сделает.

Задание 3. Разработка backend-сервиса
    Реализуем REST-сервис на FastAPI со следующими эндпоинтами:
        GET /health — проверка работоспособности сервиса (возвращает статус "ok");
        POST /predict — принимает изображение (формат: multipart/form-data с полем file) + зададим порог уверенности
        возвращает JSON:
        {
            "class_1": List[float],
            "class_2": List[float],
        }
    Обеспечим:
        валидацию входных данных (проверка формата и размера файла);
        корректную обработку ошибок (понятные сообщения об ошибке).

Задание 4. Запуск сервиса
    Сфорируем файл requirements.txt с зависимостями;
    Организуем структуру проекта (код приложения, модель, вспомогательные файлы);
    Запуск приложения через Uvicorn.
    Подготовим инструкцию для запуска сервиса локально и на удаленном сервере:
        pip install -r requirements.txt
        uvicorn main:app --reload
    Документация API: http://127.0.0.1:8000/docs

Задание 5. Тестирование и документация
    Подготовить как проект на gihub
    Можно сделать фронтенд как самая простая форма на HTML, можно использовать aiogram для бота телеграм, который
        будет получать изображение или файл, обрабатывать и отправлять ответ, как обработанное изображение с
        дополнительными данными json
    README.md:
        краткое описание задачи;
        архитектуру решения (модель, препроцессинг, структура сервиса);
        инструкции по установке зависимостей и запуску сервиса;
        пример запроса/ответа.

P.S.
    Так как у меня рабочая машина до сих пор на Win7, то версии библиотек пришлось понижать, чтоб работало.
    Проект на вид небольшой, поэтому не будем строить большую структуру проекта
P.S.S.
    Понижая версии одних библиотек, мы ломаем зависимости у других. На win7 не получилось запустить на localhost.
    Поставил все на VPS c Centos7. Но и там пришлось немного костылей поставить:
    source: https://github.com/ultralytics/yolov5/issues/1298
    Устанавливаем
    yum install xz-devel
    yum install python-backports-lzma
    pip install backports.lzma
    В папке с python
    cd /usr/local/lib/python3.10
    модифицируем файл lzma.py в импортах
    cp lzma.py lzma.py_backup_05.05.2026
    nano lzma.py
    try:
        from _lzma import *
        from _lzma import _encode_filter_properties, _decode_filter_properties
    except ImportError:
        from backports.lzma import *
        from backports.lzma import _encode_filter_properties, _decode_filter_properties
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
