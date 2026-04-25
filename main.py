from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
import json
from pathlib import Path
from swing_screener import run_full_system

app = FastAPI(title="NSE Swing Screener API")

# Path to the generated Excel file
EXCEL_PATH = Path("NSE_Swing_Screener_Report.xlsx")

@app.get("/api/scan")
async def scan_stocks(
    universe_limit: int = Query(500, ge=0),
    min_score: int = Query(6, ge=0, le=8),
    period: str = Query("2y"),
    interval: str = Query("1d"),
    top_n: int = Query(10, ge=5, le=50),
    start_index: int = Query(0, ge=0),
    symbol: str = Query(None)
):
    try:
        # Run the screener
        manual_list = [symbol] if symbol else None
        top_df = run_full_system(
            universe_limit=universe_limit,
            min_confluence_score=min_score,
            period=period,
            interval=interval,
            top_n=top_n,
            start_index=start_index,
            manual_symbols=manual_list
        )
        
        summary = getattr(top_df, "attrs", {}).get("summary", {})
        
        if top_df is None or top_df.empty:
            error = getattr(top_df, "attrs", {}).get("error", "No candidates produced.")
            content = json.dumps({"data": [], "summary": summary, "error": error})
            return Response(content=content, media_type="application/json")
            
        # Use pandas to_json which handles NaN/Inf correctly by converting them to null
        # This is more robust than manual replacement for large DataFrames
        data_json = top_df.to_json(orient="records", date_format="iso")
        summary_json = json.dumps(summary)
        
        full_response = f'{{"data": {data_json}, "summary": {summary_json}}}'
        return Response(content=full_response, media_type="application/json")
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download")
async def download_excel():
    if EXCEL_PATH.exists():
        return FileResponse(
            path=EXCEL_PATH,
            filename="NSE_Swing_Screener_Report.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    raise HTTPException(status_code=404, detail="Excel file not found. Run a scan first.")

@app.get("/api/tickers")
async def search_tickers(q: str = Query("")):
    try:
        from swing_screener import get_nse_stocks
        stocks = get_nse_stocks()
        q = q.upper()
        # Filter stocks that start with the query
        results = [s for s in stocks if q in s][:10]
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"results": []})

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
