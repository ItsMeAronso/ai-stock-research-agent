import os
import asyncio
from datetime import datetime
from typing import List
from dotenv import load_dotenv
import yfinance as yf
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# --- 1. Environment & API Configuration ---
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

llm = LLM(
    model="gemini/gemini-3.8-flash",
    temperature=0.2
)

search_tool = SerperDevTool()

# --- 2. Configurable Thresholds & Parameters ---
MIN_PRICE_THRESHOLD = float(os.getenv("MIN_PRICE_THRESHOLD", 0.0))
MIN_VOLUME_THRESHOLD = int(os.getenv("MIN_VOLUME_THRESHOLD", 0))
METRIC_HIGH_THRESHOLD = float(os.getenv("METRIC_HIGH_THRESHOLD", 1.0))
HISTORY_WINDOW = os.getenv("HISTORY_WINDOW", "30d")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))
RANKING_KEY = os.getenv("RANKING_KEY", "custom_score")

exchanges_env = os.getenv("TARGET_EXCHANGES", "")
TARGET_EXCHANGES = set(exchanges_env.split(",")) if exchanges_env else set()

HUNTER_BACKSTORY = os.getenv(
    "HUNTER_BACKSTORY",
    "An experienced market data collector focused on identifying equities with notable price action and recent news events."
)
ANALYST_BACKSTORY = os.getenv(
    "ANALYST_BACKSTORY",
    "A financial equity analyst skilled in synthesizing market news and technical metrics into clear, concise Hebrew summaries."
)


# --- 3. Strict Data Schemas (Pydantic) ---
class StockCandidate(BaseModel):
    ticker: str = Field(description="Stock ticker symbol in uppercase (e.g., ABCD)")
    headline: str = Field(description="The exact news headline")
    news_date: str = Field(description="Time/Date of the news")

class CandidateList(BaseModel):
    candidates: List[StockCandidate]

class HebrewAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    hebrew_news: str = Field(description="Concise and accurate Hebrew translation of the news")
    hebrew_analysis: str = Field(description="Brief market analysis based on news and technical metrics")

class AnalysisList(BaseModel):
    analyses: List[HebrewAnalysis]


# --- 4. Technical Verification (Python) ---
def calculate_custom_metric(current_val: int, historical_series) -> float:
    avg_val = historical_series.mean()
    return (current_val / avg_val) if avg_val > 0 else 0.0

def filter_and_verify_stocks(candidates: List[StockCandidate]) -> list:
    print("\n⚙️ Running technical verification and filtering...")
    verified_stocks = []

    for item in candidates:
        ticker = item.ticker.strip().upper()
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=HISTORY_WINDOW)
            
            if len(history) < 2:
                print(f"❌ {ticker}: Rejected (Insufficient historical data).")
                continue

            last_candle = history.iloc[-1]
            price = float(last_candle['Close'])
            volume = int(last_candle['Volume'])
            
            if price < MIN_PRICE_THRESHOLD or volume < MIN_VOLUME_THRESHOLD:
                print(f"❌ {ticker}: Rejected (Did not meet price/volume criteria).")
                continue

            info = stock.info
            exchange = info.get('exchange', 'Unknown')
            
            if TARGET_EXCHANGES and exchange not in TARGET_EXCHANGES:
                print(f"❌ {ticker}: Rejected (Traded on {exchange} - excluded exchange).")
                continue

            custom_score = calculate_custom_metric(volume, history['Volume'].iloc[:-1])
            avg_volume = history['Volume'].iloc[:-1].mean()
            
            prev_close = float(history.iloc[-2]['Close'])
            change_pct = ((price - prev_close) / prev_close) * 100

            verified_stocks.append({
                "ticker": ticker,
                "sector": info.get('sector', 'Unknown Sector'),
                "industry": info.get('industry', 'Unknown Industry'),
                "price": price,
                "change_pct": change_pct,
                "volume": volume,
                "avg_volume": int(avg_volume),
                "custom_score": custom_score,
                "metric_status": "🔥 HIGH" if custom_score > METRIC_HIGH_THRESHOLD else "❄️ NORMAL",
                "headline": item.headline,
                "news_date": item.news_date
            })
            print(f"✅ {ticker}: Verified! (Price: ${price:.2f} | Change: {change_pct:.1f}% | Score: {custom_score:.2f})")

        except Exception as e:
            print(f"⚠️ Error verifying {ticker}: {e}")

    # Sort and slice based on configurable parameters
    verified_stocks.sort(key=lambda x: x.get(RANKING_KEY, 0), reverse=True)
    return verified_stocks[:MAX_RESULTS]


# --- 5. Main Execution Flow (Async) ---
async def run_hybrid_market_scanner():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"\n📅 Starting Hybrid Market Scanner: {timestamp}")
    print("-" * 60)

    # --- Step A: AI News Hunter ---
    hunter = Agent(
        role='Market Data Collector',
        goal='Identify stocks with significant price action and recent news.',
        backstory=HUNTER_BACKSTORY,
        tools=[search_tool],
        llm=llm,
        max_iter=3,
        max_execution_time=120,
        verbose=True
    )

    hunt_task = Task(
        description=os.getenv(
            "HUNT_TASK_DESCRIPTION",
            "Search for stocks moving today due to recent news events. Return exact ticker symbols and their corresponding news headlines."
        ),
        expected_output="A structured list of stock candidates with ticker, headline, and news_date.",
        output_pydantic=CandidateList,
        agent=hunter
    )

    crew_step1 = Crew(agents=[hunter], tasks=[hunt_task], verbose=True)
    hunt_result = await crew_step1.kickoff_async()

    candidates = hunt_result.pydantic.candidates if hunt_result.pydantic else []
    if not candidates:
        print("⚠️ No matching candidates found by the agent. Exiting.")
        return

    # --- Step B: Technical Filtering & Ranking (Python) ---
    selected_stocks = filter_and_verify_stocks(candidates)
    
    if not selected_stocks:
        print("⚠️ No stocks passed the technical verification thresholds. Exiting.")
        return

    # --- Step C: AI Analyst & Translator ---
    strategist = Agent(
        role='Market Equity Analyst',
        goal='Translate stock news to Hebrew and provide a concise summary.',
        backstory=ANALYST_BACKSTORY,
        llm=llm,
        verbose=True
    )

    stocks_context = "\n".join([
        f"Ticker: {s['ticker']} | Change: {s['change_pct']:.2f}% | Score: {s['custom_score']:.2f} | News ({s['news_date']}): {s['headline']}"
        for s in selected_stocks
    ])

    analyze_task = Task(
        description=f"""
        Analyze the following validated stocks and translate their news into concise Hebrew:
        {stocks_context}
        """,
        expected_output="Structured Hebrew analysis for each validated stock.",
        output_pydantic=AnalysisList,
        agent=strategist
    )

    crew_step2 = Crew(agents=[strategist], tasks=[analyze_task], verbose=True)
    analysis_result = await crew_step2.kickoff_async()

    analyses_map = {}
    if analysis_result.pydantic:
        for item in analysis_result.pydantic.analyses:
            analyses_map[item.ticker.upper()] = item

    # --- Step D: Final Report Generation ---
    report_blocks = []
    for s in selected_stocks:
        t = s["ticker"]
        heb_data = analyses_map.get(t)
        heb_news = heb_data.hebrew_news if heb_data else s["headline"]
        heb_take = heb_data.hebrew_analysis if heb_data else "N/A"

        block = (
            f"---------------------------------------------------\n"
            f"💎 **{t}** | 🏢 **{s['sector']} ({s['industry']})**\n"
            f"💰 **מחיר:** ${s['price']:.2f} | **שינוי:** {s['change_pct']:.2f}%\n"
            f"📊 **ווליום:** {s['volume']:,} (ממוצע: {s['avg_volume']:,}) | **ציון:** {s['custom_score']:.2f} ({s['metric_status']})\n\n"
            f"📅 **זמן הדיווח:** {s['news_date']}\n"
            f"📢 **החדשות:** {heb_news}\n"
            f"🚀 **ניתוח:** {heb_take}\n"
            f"---------------------------------------------------"
        )
        report_blocks.append(block)

    final_report = "\n".join(report_blocks)

    print("\n" + "=" * 50)
    print("🔥 Market Analysis Report (Hybrid Architecture):")
    print("=" * 50)
    print(final_report)

    filename = f"market_report_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n✅ Report successfully saved to: {filename}")


# Standard script execution (compatible with both CLI and Jupyter environments)
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(run_hybrid_market_scanner())
    except RuntimeError:
        asyncio.run(run_hybrid_market_scanner())
