from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from datetime import date, datetime, timedelta
from typing import Optional, List
import bcrypt
import jwt
from jwt import InvalidTokenError
import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
SECRET_KEY    = os.getenv("SECRET_KEY", "change-this-in-production-use-a-long-random-string")
ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7 days

# ─── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./finai.db")
engine        = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal  = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base          = declarative_base()

# ─── Models ──────────────────────────────────────────────────────────────────
class UserDB(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String, nullable=False)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    expenses      = relationship("ExpenseDB", back_populates="owner", cascade="all, delete-orphan")

class ExpenseDB(Base):
    __tablename__ = "expenses"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    title       = Column(String, nullable=False)
    amount      = Column(Float, nullable=False)
    category    = Column(String, nullable=False)
    date        = Column(Date, nullable=False)
    note        = Column(String, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    owner       = relationship("UserDB", back_populates="expenses")

Base.metadata.create_all(bind=engine)

bearer = HTTPBearer()
# ─── Auth helpers ─────────────────────────────────────────────────────────────
def hash_password(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:   yield db
    finally: db.close()

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db:    Session                       = Depends(get_db)
) -> UserDB:
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ─── Schemas ──────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name:     str
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class UserOut(BaseModel):
    id:         int
    name:       str
    email:      str
    created_at: datetime
    class Config: from_attributes = True

class ExpenseCreate(BaseModel):
    title:    str
    amount:   float
    category: str
    date:     date
    note:     Optional[str] = None

class ExpenseOut(ExpenseCreate):
    id:         int
    created_at: datetime
    class Config: from_attributes = True

class ChatHistory(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistory] = []

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="FinAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.post("/api/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == req.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    user = UserDB(
        name          = req.name.strip(),
        email         = req.email.lower().strip(),
        password_hash = hash_password(req.password),
    )
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": str(user.id)})
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == req.email.lower()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user.id)})
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.get("/api/auth/me", response_model=UserOut)
def me(current_user: UserDB = Depends(get_current_user)):
    return current_user

# ─── Expense Routes ───────────────────────────────────────────────────────────
@app.get("/api/expenses", response_model=List[ExpenseOut])
def get_expenses(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(ExpenseDB).filter(ExpenseDB.user_id == current_user.id).order_by(ExpenseDB.date.desc()).all()

@app.post("/api/expenses", response_model=ExpenseOut)
def create_expense(expense: ExpenseCreate, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    db_exp = ExpenseDB(**expense.dict(), user_id=current_user.id)
    db.add(db_exp); db.commit(); db.refresh(db_exp)
    return db_exp

@app.delete("/api/expenses/{expense_id}")
def delete_expense(expense_id: int, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    exp = db.query(ExpenseDB).filter(ExpenseDB.id == expense_id, ExpenseDB.user_id == current_user.id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Expense not found")
    db.delete(exp); db.commit()
    return {"message": "Deleted"}

@app.put("/api/expenses/{expense_id}", response_model=ExpenseOut)
def update_expense(expense_id: int, expense: ExpenseCreate, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    exp = db.query(ExpenseDB).filter(ExpenseDB.id == expense_id, ExpenseDB.user_id == current_user.id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Expense not found")
    for k, v in expense.dict().items():
        setattr(exp, k, v)
    db.commit(); db.refresh(exp)
    return exp

# ─── Analytics Routes (all scoped to current user) ────────────────────────────
@app.get("/api/analytics/summary")
def get_summary(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    today         = date.today()
    cur_month     = today.strftime("%Y-%m")
    last_month    = (today.replace(day=1) - __import__('datetime').timedelta(days=1)).strftime("%Y-%m")

    def month_total(m):
        r = db.query(func.sum(ExpenseDB.amount)).filter(
            ExpenseDB.user_id == current_user.id,
            func.strftime("%Y-%m", ExpenseDB.date) == m
        ).scalar()
        return round(r or 0, 2)

    cur   = month_total(cur_month)
    last  = month_total(last_month)
    total = db.query(func.sum(ExpenseDB.amount)).filter(ExpenseDB.user_id == current_user.id).scalar() or 0
    count = db.query(func.count(ExpenseDB.id)).filter(ExpenseDB.user_id == current_user.id).scalar()

    return {
        "current_month_total": cur,
        "last_month_total":    last,
        "change_percent":      round(((cur - last) / last * 100) if last > 0 else 0, 1),
        "all_time_total":      round(total, 2),
        "expense_count":       count,
    }

@app.get("/api/analytics/monthly")
def get_monthly(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    results = db.query(
        func.strftime("%Y-%m", ExpenseDB.date).label("month"),
        func.sum(ExpenseDB.amount).label("total"),
        func.count(ExpenseDB.id).label("count")
    ).filter(ExpenseDB.user_id == current_user.id).group_by("month").order_by("month").all()
    return [{"month": r.month, "total": round(r.total, 2), "count": r.count} for r in results]

@app.get("/api/analytics/categories")
def get_categories(month: Optional[str] = None, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(ExpenseDB.category, func.sum(ExpenseDB.amount).label("total")).filter(
        ExpenseDB.user_id == current_user.id
    )
    if month:
        query = query.filter(func.strftime("%Y-%m", ExpenseDB.date) == month)
    results = query.group_by(ExpenseDB.category).all()
    return [{"category": r.category, "total": round(r.total, 2)} for r in results]

# ─── AI Chat ──────────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    expenses   = db.query(ExpenseDB).filter(ExpenseDB.user_id == current_user.id).order_by(ExpenseDB.date.desc()).limit(100).all()
    summary    = get_summary(current_user, db)
    monthly    = get_monthly(current_user, db)
    categories = get_categories(db=db, current_user=current_user)

    expense_data = [{"title": e.title, "amount": e.amount, "category": e.category, "date": str(e.date), "note": e.note} for e in expenses]

    system = f"""You are FinAI, a sharp and friendly personal finance manager for {current_user.name}.
You have complete access to their expense data and give actionable, data-driven financial advice.

## {current_user.name}'s Financial Snapshot:
- This month's spending: ₹{summary['current_month_total']:,.2f}
- Last month's spending: ₹{summary['last_month_total']:,.2f}
- Month-over-month change: {summary['change_percent']}%
- All-time total tracked: ₹{summary['all_time_total']:,.2f}
- Total expenses logged: {summary['expense_count']}

## Monthly Trends (last 6):
{json.dumps(monthly[-6:], indent=2)}

## Spending by Category:
{json.dumps(categories, indent=2)}

## Recent Transactions:
{json.dumps(expense_data[:50], indent=2)}

## Instructions:
- Address the user by their first name occasionally
- Answer with specific numbers from their data
- Give actionable, personalized budgeting advice
- Identify overspending patterns and suggest specific cuts
- Use ₹ (Indian Rupee) symbol
- Keep responses concise: 2-3 paragraphs max
- Be warm, smart, and encouraging"""

    messages = [{"role": h.role, "content": h.content} for h in req.history]
    messages.append({"role": "user", "content": req.message})

    client   = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=600,
        temperature=0.7,
    )
    return {"reply": response.choices[0].message.content}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}
