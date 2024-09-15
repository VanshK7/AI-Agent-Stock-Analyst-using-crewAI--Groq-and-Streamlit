# Stock Researcher & Analyst AI Agent

This repository contains an AI-powered financial analysis system that assists users in researching stocks, analyzing financial data, assessing risks, performing technical and fundamental analysis, and fetching the latest news. It uses **LangChain**, **yfinance**, **Streamlit**, and **CrewAI** with **LLama 70B** models to provide insightful financial reports.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Usage](#usage)

## Overview

This project creates multiple AI agents that collaborate to gather stock information, perform technical/fundamental analysis, evaluate risk, and provide detailed stock reports. These agents use the **LLama 70B** model to dynamically interact and generate financial insights tailored to user queries.

## Features

- Fetch basic stock information (name, sector, market cap, etc.)
- Perform fundamental analysis (P/E ratio, EPS, revenue growth, etc.)
- Conduct risk assessments (volatility, beta, Sharpe ratio, etc.)
- Perform technical analysis (SMA, RSI, MACD, trend, etc.)
- Fetch recent news articles for stocks
- Combine all analyses into comprehensive stock reports

## Technologies Used

- **LangChain**
- **CrewAI**
- **yfinance**
- **Streamlit**
- **LLama3 70B** from **Groq**

## Usage

To use the Stock Researcher & Financial Analysis AI Agents, follow these steps:

### Running the Streamlit App

1. **Start the Streamlit App**:
   After setting up your environment and installing dependencies, launch the app using:
   
   `streamlit run main.py`