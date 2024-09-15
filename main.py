import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from crewai import Crew, Agent, Task, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
import numpy as np
import streamlit as st


GROQ_API_KEY = "YOUR_GROQ_API_KEY"

llm_llama70b = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

@tool
def get_basic_stock_info(ticker: str) -> pd.DataFrame:
    """Retrieves basic information about a single stock.
    For more information, you can perform technical analysis, assess stock risk or perform fundamental analysis.
    
    Params:
    - ticker: The stock ticker symbol.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    basic_info = pd.DataFrame({
        'Name': [info.get('longName', 'N/A')],
        'Sector': [info.get('sector', 'N/A')],
        'Industry': [info.get('industry', 'N/A')],
        'Market Cap': [info.get('marketCap', 'N/A')],
        'Current Price': [info.get('currentPrice', 'N/A')],
        '52 Week High': [info.get('fiftyTwoWeekHigh', 'N/A')],
        '52 Week Low': [info.get('fiftyTwoWeekLow', 'N/A')],
        'Average Volume': [info.get('averageVolume', 'N/A')]
    })
    return basic_info

@tool
def get_fundamental_analysis(ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Performs fundamental analysis on a given stock for a specific period.
    
    Params:
    - ticker: The stock ticker symbol.
    - period: The period to consider for historical data (default is 1 year).
    
    Returns: 
    - DataFrame with fundamental metrics.
    """
    stock = yf.Ticker(ticker)
    
    # Fetch historical data for the given period
    history = stock.history(period=period)
    
    # Fetch latest available financial info
    info = stock.info
    
    fundamental_analysis = pd.DataFrame({
        'PE Ratio': [info.get('trailingPE', 'N/A')],
        'Forward PE': [info.get('forwardPE', 'N/A')],
        'PEG Ratio': [info.get('pegRatio', 'N/A')],
        'Price to Book': [info.get('priceToBook', 'N/A')],
        'Dividend Yield': [info.get('dividendYield', 'N/A')],
        'EPS (TTM)': [info.get('trailingEps', 'N/A')],
        'Revenue Growth': [info.get('revenueGrowth', 'N/A')],
        'Profit Margin': [info.get('profitMargins', 'N/A')],
        'Free Cash Flow': [info.get('freeCashflow', 'N/A')],
        'Debt to Equity': [info.get('debtToEquity', 'N/A')],
        'Return on Equity': [info.get('returnOnEquity', 'N/A')],
        'Operating Margin': [info.get('operatingMargins', 'N/A')],
        'Quick Ratio': [info.get('quickRatio', 'N/A')],
        'Current Ratio': [info.get('currentRatio', 'N/A')],
        'Earnings Growth': [info.get('earningsGrowth', 'N/A')],
        'Stock Price Avg (Period)': [history['Close'].mean()],
        'Stock Price Max (Period)': [history['Close'].max()],
        'Stock Price Min (Period)': [history['Close'].min()]
    })
    
    return fundamental_analysis

@tool
def get_stock_risk_assessment(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Performs a risk assessment on a given stock.
    
    Params:
    - ticker: The stock ticker symbol.
    - period: The time period for historical data (default: "1y").
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    
    # Calculate daily returns
    returns = history['Close'].pct_change().dropna()
    
    # Calculate risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    beta = calculate_beta(returns, '^GSPC', period)  # Beta relative to S&P 500
    var_95 = np.percentile(returns, 5)  # 95% Value at Risk
    max_drawdown = calculate_max_drawdown(history['Close'])
    
    risk_assessment = pd.DataFrame({
        'Annualized Volatility': [volatility],
        'Beta': [beta],
        'Value at Risk (95%)': [var_95],
        'Maximum Drawdown': [max_drawdown],
        'Sharpe Ratio': [calculate_sharpe_ratio(returns)],
        'Sortino Ratio': [calculate_sortino_ratio(returns)]
    })
    
    return risk_assessment

def calculate_beta(stock_returns, market_ticker, period):
    market = yf.Ticker(market_ticker)
    market_history = market.history(period=period)
    market_returns = market_history['Close'].pct_change().dropna()
    
    # Align the dates of stock and market returns
    aligned_returns = pd.concat([stock_returns, market_returns], axis=1).dropna()
    
    covariance = aligned_returns.cov().iloc[0, 1]
    market_variance = market_returns.var()
    
    return covariance / market_variance

def calculate_max_drawdown(prices):
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

@tool
def get_technical_analysis(ticker: str, period: str = "") -> pd.DataFrame:
    """Perform technical analysis on a given stock.
    
    Params:
    - ticker: The stock ticker symbol.
    - period: The time period for historical data (available time-periods: ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]).
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    
    # Calculate indicators
    history['SMA_50'] = history['Close'].rolling(window=50).mean()
    history['SMA_200'] = history['Close'].rolling(window=200).mean()
    history['RSI'] = calculate_rsi(history['Close'])
    history['MACD'], history['Signal'] = calculate_macd(history['Close'])
    
    latest = history.iloc[-1]
    
    analysis = pd.DataFrame({
        'Indicator': [
            'Current Price',
            '50-day SMA',
            '200-day SMA',
            'RSI (14-day)',
            'MACD',
            'MACD Signal',
            'Trend',
            'MACD Signal',
            'RSI Signal'
        ],
        'Value': [
            f'${latest["Close"]:.2f}',
            f'${latest["SMA_50"]:.2f}',
            f'${latest["SMA_200"]:.2f}',
            f'{latest["RSI"]:.2f}',
            f'{latest["MACD"]:.2f}',
            f'{latest["Signal"]:.2f}',
            analyze_trend(latest),
            analyze_macd(latest),
            analyze_rsi(latest)
        ]
    })
    
    return analysis

@tool
def get_stock_news(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Fetches recent news articles related to a specific stock.
    
    Params:
    - ticker: The stock ticker symbol.
    - limit: The number of news articles to fetch.
    """
    stock = yf.Ticker(ticker)
    news = stock.news[:limit]
    
    news_data = []
    for article in news:
        news_entry = {
            "Title": article['title'],
            "Publisher": article['publisher'],
            "Published": datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S'),
            "Link": article['link']
        }
        news_data.append(news_entry)
    
    return pd.DataFrame(news_data)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def analyze_trend(latest):
    if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
        return "Bullish"
    elif latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
        return "Bearish"
    else:
        return "Neutral"

def analyze_macd(latest):
    if latest['MACD'] > latest['Signal']:
        return "Bullish"
    else:
        return "Bearish"

def analyze_rsi(latest):
    if latest['RSI'] > 70:
        return "Overbought"
    elif latest['RSI'] < 30:
        return "Oversold"
    else:
        return "Neutral"

def analyze_bollinger_bands(latest):
    if latest['Close'] > latest['BB_Upper']:
        return "Price above upper band (potential overbought)"
    elif latest['Close'] < latest['BB_Lower']:
        return "Price below lower band (potential oversold)"
    else:
        return "Price within bands"

def format_number(self, value):
    if value != 'N/A':
        return f'${value:,.2f}'
    else:
        return 'N/A'

def interpret_pe_ratio(self, trailing_pe):
    if trailing_pe < 15:
        return "Undervalued"
    elif trailing_pe > 30:
        return "Overvalued"
    else:
        return "Neutral"

def interpret_price_to_book(self, price_to_book):
    if price_to_book < 1:
        return "Undervalued"
    elif price_to_book > 3:
        return "Overvalued"
    else:
        return "Neutral"
    
    
stock_researcher = Agent(
    llm=llm_llama70b,
    role="Stock Researcher",
    goal="Identify the stock and the stock ticker, and if you already have the stock ticker and if it's necessary, get basic stock info about the selected stock.",
    backstory="An junior stock researcher with a knack for gathering relevant, basic information about stocks, the relevant company/companies, the industry, and some basic info about stock's performance",
    tools=[get_basic_stock_info],
    verbose=True,
    allow_delegation=False
)

financial_analyst = Agent(
    llm=llm_llama70b,
    role="Financial Analyst",
    goal="Perform in-depth fundamental and technical analysis on the stock, focusing on aspects most relevant to the user's query",
    backstory="A seasoned financial analyst with expertise in interpreting complex financial data and translating it into insights tailored to various levels of financial literacy",
    tools=[get_technical_analysis, get_fundamental_analysis, get_stock_risk_assessment],
    verbose=True,
    allow_delegation=False
)

news_analyst = Agent(
    llm=llm_llama70b,
    role="News Analyst",
    goal="Fetch recent news articles related to the stock and their potential impact on performance",
    backstory="A sharp news analyst who can quickly digest information, assess its relevance to stock performance, and provide concise summaries",
    tools=[get_stock_news],
    verbose=True
)

report_writer = Agent(
    role='Financial Report Writer',
    goal='Synthesize all analysis into a cohesive, professional stock report',
    backstory='Experienced financial writer with a talent for clear, concise reporting',
    tools=[],
    verbose=True,
    allow_delegation=False,
    llm=llm_llama70b
)

collect_stock_info = Task(
    description='''
    1. Extract the ticker of the stock (or stocks) mentioned in the user query as well as the timeframe (if mentioned). If the ticker is not provided, use the query to identify the stock ticker.
    2. If the query implies a novice user, prepare brief explanations for key financial terms. If nothing is mentioned, assume that the user has an above average understanding of financial terms.
    
    Expect only basic stock info from this task.
    
    User query: {query}.
    
    Your response should be on the basis of:
    Ticker: [identified stock ticker]
    Timeframe: [identified timeframe]
    Analysis Focus: [identified focus of analysis]
    User Expertise: [implied level of financial expertise]
    Key Concerns: [specific concerns or priorities mentioned]
    ''',
    expected_output="A summary of the stock's key financial metrics and performance, tailored to the user's query.",
    agent=stock_researcher,
    dependencies=[],
    context=[]
)

perform_analysis = Task(
    description='''
    Conduct a thorough analysis of the stock, tailored to the user's query and expertise level.
    1. Use the get_stock_info, get_fundamental_analysis, get_stock_risk_assessment and get_technical_analysis tools as needed, based on the query's focus. E.g. If the query is about the fundamentals of a stock, then technical info need not be present.
    2. Focus on metrics and trends most relevant to the user's specific question and identified timeframe.
    3. Provide clear explanations of complex financial concepts if the query suggests a novice user.
    4. Relate the analysis directly to the key concerns identified in the query interpretation.
    5. Consider both historical performance and future projections in your analysis..
    
    User query: {query}.
    ''',
    expected_output="A detailed analysis of the stock's financial and/or technical performance, directly addressing the user's query and concerns.",
    agent=financial_analyst,
    dependencies=[collect_stock_info],
    context=[collect_stock_info]
)

analyze_stock_news = Task(
    description='''
    1. Use the get_stock_news tool to fetch recent news related to the stock.
    2. Conclude with an overall assessment including the sentiment of how recent news might influence the stock in the relevant timeframe.
    
    NOTE: Re-fetching news will get you the same results.
    ''',
    expected_output="A summary of recent news articles related to the stock and their potential impact on performance.",
    agent=news_analyst,
    dependencies=[collect_stock_info],
    context=[collect_stock_info]
)

generate_stock_report = Task(
    description='''
    Synthesize all the collected information and analyses into a stock report tailored to the user's specific query.
    The report should:
    1. Begin with an Executive Summary that directly addresses the user's question
    2. Include relevant sections based on the query's focus
    3. Provide an Investment Recommendation that specifically answers the user's query
    4. Conclude with a summary that ties all insights back to the original question

    Ensure that:
    - The report directly answers the user's specific question
    - The language and depth of analysis match the user's level of expertise implied by the query
    - The report highlights factors most relevant to the user's identified concerns and timeframe
    - Clear, professional language is used throughout, with well-reasoned insights
    - The report is in markdown format for easy reading and formatting
    - The report should be crisp but detailed. You can reiterate important points but avoid redundancy.
    - You are an expert in the field, so you should be confident in your answer, requiring no further action/analysis from the user. It is your job to give a clear recommendation.
    - The report should contain only the relevant info. E.g. if the query is about the fundamentals of a stock, then technical info need not be present.
    
    User query: {query}.
    ''',
    expected_output="A comprehensive stock report in markdown format, addressing all aspects of the user's query and providing a clear investment recommendation.",
    agent=report_writer,
    dependencies=[collect_stock_info],
    context=[collect_stock_info, perform_analysis, analyze_stock_news]
)

crew = Crew(
    agents=[stock_researcher, financial_analyst, news_analyst, report_writer],
    tasks=[
        collect_stock_info,
        perform_analysis,
        analyze_stock_news,
        generate_stock_report
    ],
    process=Process.sequential,
    manager_llm=llm_llama70b
)

st.set_page_config(page_title="Advanced Stock Analysis Dashboard", layout="wide")

st.title("Advanced Stock Analysis Dashboard")

st.sidebar.header("Stock Analysis Query")
query = st.sidebar.text_area("Enter your stock analysis question", value="Is Apple a safe long-term bet for a risk-averse individual?", height=100)
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    st.info(f"Starting analysis for query: {query}. This may take a few minutes...")

    default_date = datetime.now().date()
    result = crew.kickoff(inputs={"query": query, "default_date": str(default_date)})
    
    st.success("Analysis complete!")
    
    st.markdown("## Full Analysis Report")
    st.markdown(result)

st.markdown("---")
st.markdown("Made by Vansh Kharidia")
