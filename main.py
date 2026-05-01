import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from groq import Groq
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "reportai.db"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-insecure-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

if JWT_SECRET_KEY == "dev-insecure-change-me":
    logger.warning(
        "Используется JWT_SECRET_KEY по умолчанию; задайте JWT_SECRET_KEY в .env для продакшена"
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
http_bearer = HTTPBearer()

app = FastAPI(title="Report AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent), name="static")
_groq_client: Optional[Groq] = None


class MarketingPayload(BaseModel):
    sessions: int = Field(..., ge=0)
    conversions: int = Field(..., ge=0)
    ad_spend: float = Field(..., ge=0)
    revenue: float = Field(..., ge=0)
    period: str = Field(..., min_length=1)


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginBody(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
        logger.info("База SQLite инициализирована: %s", DB_PATH)
    finally:
        conn.close()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_access_token(subject_email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": subject_email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_email(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(http_bearer)],
) -> str:
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        email = payload.get("sub")
        if not email or not isinstance(email, str):
            raise HTTPException(
                status_code=401,
                detail="Недействительный токен",
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Недействительный токен",
        ) from None


@app.on_event("startup")
def startup() -> None:
    global _groq_client
    init_db()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        _groq_client = None
        logger.warning("GROQ_API_KEY не задан в окружении")
    else:
        _groq_client = Groq(api_key=key)
        logger.info("Клиент Groq настроен")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/register")
def register(body: RegisterBody) -> dict[str, str]:
    email = body.email.strip().lower()
    password_hash = pwd_context.hash(body.password)
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        logger.info("Попытка регистрации с занятым email: %s", email)
        raise HTTPException(status_code=409, detail="Этот email уже зарегистрирован")
    finally:
        conn.close()

    logger.info("Зарегистрирован пользователь: %s", email)
    return {"email": email, "message": "Регистрация успешна"}


@app.post("/login")
def login(body: LoginBody) -> dict[str, str]:
    email = body.email.strip().lower()
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    finally:
        conn.close()

    if row is None or not pwd_context.verify(body.password, row["password_hash"]):
        logger.info("Неудачный вход: %s", email)
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_access_token(email)
    logger.info("Успешный вход: %s", email)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/generate-report")
def generate_report(
    payload: MarketingPayload,
    _email: Annotated[str, Depends(get_current_email)],
) -> dict[str, str]:
    if _groq_client is None:
        logger.error("Запрос отклонён: отсутствует GROQ_API_KEY")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY не настроен")

    conv_rate = (
        (payload.conversions / payload.sessions * 100) if payload.sessions else 0.0
    )
    roas = (payload.revenue / payload.ad_spend) if payload.ad_spend else 0.0

    prompt = f"""Ты опытный маркетолог. Напиши один связный абзац на русском языке для клиента:
объясни результаты периода без канцелярита, дружелюбно и по делу.

Данные:
- Период: {payload.period}
- Визиты (sessions): {payload.sessions}
- Конверсии: {payload.conversions}
- Конверсия в визит: {conv_rate:.2f}%
- Рекламный бюджет (USD): {payload.ad_spend:.2f}
- Доход (USD): {payload.revenue:.2f}
- ROAS (доход/расход): {roas:.2f}

Не используй маркированные списки — только один абзац текста."""

    try:
        completion = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.exception("Ошибка вызова Groq: %s", e)
        raise HTTPException(status_code=502, detail="Ошибка генерации отчёта") from e

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        logger.error("Groq вернул пустой ответ")
        raise HTTPException(status_code=502, detail="Пустой ответ модели")

    logger.info("Отчёт успешно сгенерирован для периода: %s", payload.period)
    return {"report": text}
