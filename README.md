# AI Stock Research Agent (Hybrid Architecture)

An automated, AI-driven market scanning system designed to identify equities experiencing significant price action alongside fresh news events.

## System Architecture
1. **Data Collection (AI Agent):** Scans real-time financial news sources using `CrewAI` and `SerperDevTool`.
2. **Technical Verification (Python):** Validates candidates against historical and live market data (`yfinance`).
3. **Synthesis & Localization (AI Agent):** Translates validated news catalysts into Hebrew and generates a concise market summary.

## Setup & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/ItsMeAronso/ai-stock-research-agent.git](https://github.com/ItsMeAronso/ai-stock-research-agent.git)
   cd ai-stock-research-agent