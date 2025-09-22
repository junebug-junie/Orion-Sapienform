from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from conjourney_sdk import collapse_log
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()
print("POSTGRES_URI:", os.getenv("POSTGRES_URI"))

# Optional: CORS setup if SDK/frontend will call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount collapse mirror logging routes
app.include_router(collapse_log.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Conjourney Memory API is alive"}

# Optional: run block if launching via `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8086, reload=True)
