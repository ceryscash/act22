from flask import Flask, render_template, request, jsonify
from litellm import completion
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)


class StockAnalyzer:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol.upper()

    def get_stock_data(self):
        stock = yf.Ticker(self.stock_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        return stock.history(start=start_date, end=end_date)

    def create_prompt(self, role, stock_data):
        last_price = stock_data['Close'].iloc[-1]
        high_52w = stock_data['High'].max()
        low_52w = stock_data['Low'].min()

        prompts = {
            'researcher': f"""
                You're a seasoned researcher with a knack for predicting the next upcoming trends in stocks. 
                Known for your ability to find the most relevant information and analyze it in a clear and concise manner. 
                Pull relevant information (price history, volume, and news) from Yahoo Finance about {self.stock_symbol}'s trends and predictions. 
                Focus on: price history, volume, and news. 
                Current: ${last_price:.2f}, 52W: ${high_52w:.2f}-${low_52w:.2f}. 
                Keep under 200 words.
            """,

            'accountant': f"""
                You’re an experienced financial analyst focused on assessing the financial health and potential growth of {self.stock_symbol}. 
                Calculate and interpret key accounting ratios for financial analysis, specializing in liquidity (current ratio, quick ratio), 
                profitability (gross margin, ROA, ROE), growth (revenue and EPS growth), and valuation (P/E ratio, EV/EBITDA). 
                These insights help in determining the financial viability of the stock. 
                Current: ${last_price:.2f}, 52W: ${high_52w:.2f}-${low_52w:.2f}. 
                Keep under 200 words. This advice will not be used, just make suggestions baseed on trends whether you think decisions would be a loss or a profit.
            """,

            'recommender': f"""
                You’re a strategic analyst responsible for making conclusive recommendations by integrating insights from both the Researcher and Accountant. 
                By applying a scoring system and using machine learning models, you can optimize portfolios and assess risk. 
                Your decisions are shaped by both quantitative scores and qualitative inputs to provide balanced and clear investment advice about the stock: 
                {self.stock_symbol}. Use researched data and financial analysis to generate actionable stock recommendations (buy, sell). 
                Current: ${last_price:.2f}, 52W: ${high_52w:.2f}-${low_52w:.2f}. 
                Keep under 200 words.
            """,

            'blogger': f"""
                You’re a meticulous reporting analyst known for turning complex data into digestible reports, allowing clients to easily interpret and act on recommendations. 
                You summarize the findings, breaking down key elements into sections, making the data accessible and easy to understand using concise explanations and accessible vocabulary. 
                Create detailed, structured reports summarizing stock data about {self.stock_symbol}, including analysis and recommendations. 
                Current: ${last_price:.2f}, 52W: ${high_52w:.2f}-${low_52w:.2f}. 
                Keep under 200 words.
            """
        }

        return prompts.get(role)

    def query_llm(self, prompt):
        response = completion(
            model="ollama/llama3.2:1b",
            messages=[{"content": prompt, "role": "user"}]
        )
        return response['choices'][0]['message']['content']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        stock_symbol = request.json.get('stock_symbol')
        if not stock_symbol:
            return jsonify({'success': False, 'error': 'Stock symbol is required'})

        analyzer = StockAnalyzer(stock_symbol)
        stock_data = analyzer.get_stock_data()

        results = {}
        for role in ['researcher', 'accountant', 'recommender', 'blogger']:
            prompt = analyzer.create_prompt(role, stock_data)
            results[role] = analyzer.query_llm(prompt)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)