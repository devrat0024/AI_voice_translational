import uvicorn
import os

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENV", "development") == "development"
    
    print(f"Starting FastAPI server on http://{host}:{port} ...")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
