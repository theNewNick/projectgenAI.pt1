import os
import io
import time
import asyncio
import logging
import openai
from openai.error import APIError, RateLimitError
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
from flask import Flask, request, send_file, render_template, jsonify
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from asgiref.wsgi import WsgiToAsgi  # For ASGI compatibility

# Set up Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Apply ProxyFix middleware to handle HTTPS behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logging
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
openai.api_key = openai_api_key

# Set your Flask secret key
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'YOUR_FLASK_SECRET_KEY')

# Directory to save uploaded files (ensure this exists)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update your Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'csv': {'csv'},
    'pdf': {'pdf'},
    'image': {'png', 'jpg', 'jpeg', 'gif'}
}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# DCF and Ratio Analysis Functions
def dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years):
    """Discounted Cash Flow analysis"""
    logger.debug('Starting DCF analysis.')
    discounted_free_cash_flows = [
        fcf / (1 + wacc) ** i for i, fcf in enumerate(projected_free_cash_flows, 1)
    ]
    discounted_terminal_value = terminal_value / (1 + wacc) ** projection_years
    dcf_value = sum(discounted_free_cash_flows) + discounted_terminal_value
    logger.debug('Completed DCF analysis.')
    return dcf_value

def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning None if denominator is zero."""
    try:
        result = numerator / denominator if denominator != 0 else None
        logger.debug(f'Safe divide {numerator} / {denominator} = {result}')
        return result
    except Exception as e:
        logger.error(f'Error in safe_divide: {str(e)}')
        return None

def calculate_ratios(financials, benchmarks):
    """Calculate key financial ratios and compare them to benchmarks."""
    logger.debug('Starting ratio analysis.')
    # Liquidity Ratios
    current_ratio = safe_divide(financials['current_assets'], financials['current_liabilities'])

    # Leverage Ratios
    debt_to_equity = safe_divide(financials['total_debt'], financials['shareholders_equity'])

    # Valuation Ratios
    pe_ratio = safe_divide(financials['market_price'], financials['eps'])
    pb_ratio = safe_divide(financials['market_price'], financials['book_value_per_share'])

    # Compare with benchmarks and assign scores
    scores = {}

    # Debt-to-Equity Ratio
    if debt_to_equity is not None:
        scores['debt_to_equity'] = 1 if debt_to_equity < benchmarks['debt_to_equity'] else -1
    else:
        scores['debt_to_equity'] = 0

    # Current Ratio
    if current_ratio is not None:
        scores['current_ratio'] = 1 if current_ratio > benchmarks['current_ratio'] else -1
    else:
        scores['current_ratio'] = 0

    # P/E Ratio
    if pe_ratio is not None:
        scores['pe_ratio'] = 1 if pe_ratio < benchmarks['pe_ratio'] else -1
    else:
        scores['pe_ratio'] = 0

    # P/B Ratio
    if pb_ratio is not None:
        scores['pb_ratio'] = 1 if pb_ratio < benchmarks['pb_ratio'] else -1
    else:
        scores['pb_ratio'] = 0

    # Aggregate the scores
    total_score = sum(scores.values())
    logger.debug(f'Ratio analysis scores: {scores}, Total Score: {total_score}')

    # Normalize the total score to -1, 0, or 1
    if total_score >= 2:
        normalized_factor2_score = 1
    elif total_score <= -2:
        normalized_factor2_score = -1
    else:
        normalized_factor2_score = 0

    logger.debug(f'Normalized Factor 2 Score: {normalized_factor2_score}')

    return {
        'Current Ratio': current_ratio,
        'Debt-to-Equity Ratio': debt_to_equity,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Scores': scores,
        'Total Score': total_score,
        'Normalized Factor 2 Score': normalized_factor2_score #Add the normalized score here
    }

def calculate_cagr(beginning_value, ending_value, periods):
    """Calculate the Compound Annual Growth Rate."""
    try:
        if beginning_value <= 0 or periods <= 0:
            logger.warning('Invalid beginning value or periods for CAGR calculation.')
            return None
        cagr = (ending_value / beginning_value) ** (1 / periods) - 1
        logger.debug(f'Calculated CAGR: {cagr}')
        return cagr
    except Exception as e:
        logger.error(f'Error calculating CAGR: {str(e)}')
        return None

def generate_plot(x, y, title, x_label, y_label):
    logger.debug(f'Generating plot: {title}')
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG')
    plt.close()
    img_data.seek(0)
    logger.debug(f'Plot generated: {title}')
    return img_data

def generate_benchmark_comparison_plot(benchmark_comparison):
    labels = benchmark_comparison['Ratios']
    company_values = benchmark_comparison['Company']
    industry_values = benchmark_comparison['Industry']

    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the bars
    bars_company = ax.bar(x - width/2, company_values, width, label='Company')
    bars_industry = ax.bar(x + width/2, industry_values, width, label='Industry')

    # Add labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Ratio Values')
    ax.set_title('Company vs. Industry Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Function to attach a text label above each bar displaying its height
    def autolabel(rects):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars_company)
    autolabel(bars_industry)

    fig.tight_layout()

    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG', bbox_inches='tight')
    plt.close(fig)
    img_data.seek(0)
    return img_data


# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    logger.debug(f'Extracting text from PDF: {file_path}')
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num, page in enumerate(reader.pages, 1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                else:
                    logger.warning(f'No text found on page {page_num} of {file_path}.')
            if not text.strip():
                logger.warning(f'No extractable text found in {file_path}.')
                return None
            logger.debug(f'Text extracted from PDF: {file_path}')
            return text
    except Exception as e:
        logger.error(f'Error reading PDF file {file_path}: {str(e)}')
        return None

# Define a semaphore to limit the number of concurrent API calls
semaphore = asyncio.Semaphore(5)  # Adjust this based on OpenAI's rate limits

# Asynchronous function to call OpenAI summarization API
async def call_openai_summarization(text):
    retry_delay = 5  # Initial delay
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await openai.ChatCompletion.acreate(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                max_tokens=500,
                n=1,
                temperature=0.5,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except RateLimitError:
            logger.warning(f'Rate limit error, retrying in {retry_delay} seconds...')
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except APIError as e:
            logger.error(f'OpenAI API error: {str(e)}')
            return None
        except Exception as e:
            logger.error(f'Unexpected error during OpenAI API call: {str(e)}')
            return None
    logger.error('Failed to summarize text after multiple attempts.')
    return None

# Asynchronous function to call GPT-3.5 for company, industry, and risk considerations
async def generate_summaries_and_risks(ten_k_text):
    logger.debug('Generating company, industry summaries, and risk considerations from 10-K report.')

    # Define separate prompts for each section
    company_prompt = f"""
Summarize the company based on the following 10-K report, focusing on the following key areas. 
Keep the summary under 500 words and prioritize concise, factual information:

1. **Company Name**: Mention the official company name.
2. **Business Overview**: Provide a brief description of what the company does, including its products or services and the industry it operates in.
3. **Key Personnel**: List the names and titles of key executives or management (e.g., CEO, CFO).
4. **Products and Services**: Summarize the main products or services the company offers.
5. **Target Market**: Describe the company’s customer base or target audience.
6. **Market Position**: Highlight the company’s competitive advantage or key differentiators in the market.
7. **Financial Overview (High-Level)**: Provide key financial metrics like revenue, profitability, or recent growth trends.
8. **Future Plans or Strategic Goals**: Briefly outline the company’s future growth plans or strategic direction.

If any of the above information is not available in the report, omit that section.

Here is the 10-K report:

{ten_k_text}
"""
    industry_prompt = f"""
Summarize the industry based on the following 10-K report, focusing on the following key areas. 
Keep the summary under 500 words and prioritize concise, factual information:

1. **Industry Overview**: Provide a brief summary of the industry, including key sectors and the scope of the report.
2. **Market Size and Growth**: Mention the current market size, growth trends, and any revenue, market share, or demand projections.
3. **Key Players**: List major companies or competitors within the industry.
4. **Market Trends**: Identify emerging trends, technological advancements, or changes shaping the industry.
5. **Competitive Landscape**: Analyze the competition, highlighting market leaders, challengers, and any disruptive companies.
6. **Regulatory Environment**: Summarize relevant regulations, policies, or government actions impacting the industry.
7. **Opportunities**: Describe areas of potential growth, expansion, or innovation within the industry.
8. **Challenges**: Mention key challenges or barriers to entry, such as regulatory hurdles, high competition, or resource scarcity.

If any of the above information is not available in the report, omit that section.

Here is the relevant section of the 10-K report:

{ten_k_text}
"""
    risks_prompt = f"""
Identify and summarize the key risk considerations from the following 10-K report, focusing on the following categories. 
Keep the summary under 500 words and prioritize concise, factual information:

1. **Market Risks**: Summarize risks related to market demand, competition, and changes in the industry that could impact revenue or the company's market position.
2. **Operational Risks**: Highlight internal risks such as supply chain disruptions, production issues, or management inefficiencies.
3. **Financial Risks**: Describe risks associated with liquidity, debt, currency fluctuations, or access to capital.
4. **Regulatory and Compliance Risks**: Identify potential risks from changing regulations, legal requirements, or non-compliance penalties.
5. **Technological Risks**: Summarize risks related to technological changes, cybersecurity threats, or failure to adapt to new technology.
6. **Reputation Risks**: Highlight risks to the company’s public image due to poor performance, scandals, or negative media coverage.
7. **Environmental and Social Risks**: Identify risks related to sustainability, environmental regulations, or social responsibility that could affect operations or public perception.

If any of the above risk categories are not addressed in the 10-K report, omit that section from the summary.

Here is the relevant section of the 10-K report:

{ten_k_text}
"""

    try:
        # Define a helper function to handle chunking
        async def summarize_with_token_check(prompt, max_tokens=500):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            max_context_length = 7000  # Max tokens for gpt-3.5-turbo

            # Encode and check the length of the tokens
            tokens = encoding.encode(prompt)
            if len(tokens) > max_context_length - max_tokens:
                logger.warning('Prompt is too long, breaking it into smaller chunks for summarization.')

                # Split the text into chunks
                chunk_size = max_context_length - max_tokens
                chunks = [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

                # Summarize each chunk and combine the summaries
                summaries = []
                for chunk in chunks:
                    summary = await call_openai_summarization(chunk)
                    if summary:
                        summaries.append(summary)
                return " ".join(summaries)  # Combine the chunk summaries

            # If the prompt fits within the token limit, summarize it directly
            return await call_openai_summarization(prompt)

        # Asynchronously call GPT-3.5 for each summary
        company_summary_task = summarize_with_token_check(company_prompt)
        industry_summary_task = summarize_with_token_check(industry_prompt)
        risks_task = summarize_with_token_check(risks_prompt)

        # Gather the results
        company_summary, industry_summary, risks_summary = await asyncio.gather(
            company_summary_task, industry_summary_task, risks_task
        )

        logger.debug('Company, industry summaries, and risk considerations generated successfully.')

        return company_summary, industry_summary, risks_summary

    except Exception as e:
        logger.error(f"Error generating summaries and risks from 10-K report: {str(e)}")
        return None, None, None


# Asynchronous function to call OpenAI sentiment analysis API
async def call_openai_analyze_sentiment(text, context):
    retry_delay = 5  # Initial delay
    max_retries = 5
    prompt = f"""
As an expert financial analyst, analyze the following {context}.
Provide a sentiment score between -1 (very negative) and 1 (very positive).
Also, briefly explain the main factors contributing to this sentiment.

Text:
{text}

Response Format:
Sentiment Score: [score]
Explanation: [brief explanation]
"""
    for attempt in range(max_retries):
        try:
            logger.debug(f'Attempting sentiment analysis, attempt {attempt+1}.')
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in sentiment analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                temperature=0.5,
            )
            content = response.choices[0].message.content.strip()
            # Extract the sentiment score and explanation from the response
            lines = content.split('\n')
            sentiment_score = None
            explanation = ""
            for line in lines:
                if "Sentiment Score:" in line:
                    try:
                        sentiment_score = float(line.split("Sentiment Score:")[1].strip())
                    except ValueError:
                        sentiment_score = None
                elif "Explanation:" in line:
                    explanation = line.split("Explanation:", 1)[1].strip()
            return sentiment_score, explanation
        except RateLimitError:
            logger.warning(f'Rate limit exceeded during sentiment analysis. Retrying in {retry_delay} seconds...')
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except APIError as e:
            logger.error(f'OpenAI API error during sentiment analysis: {str(e)}')
            return None, ""
        except Exception as e:
            logger.error(f'Error performing sentiment analysis: {str(e)}')
            return None, ""
    logger.error('Failed to perform sentiment analysis after multiple attempts.')
    return None, ""

# Asynchronous function to summarize text
async def summarize_text_async(text):
    logger.debug('Starting text summarization.')
    max_tokens_per_chunk = 3800  # Adjusted to 3800 tokens for safety
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Tokenize the text
    tokens = encoding.encode(text)
    token_count = len(tokens)

    # If text is within limits, summarize directly
    if token_count <= max_tokens_per_chunk:
        summary = await call_openai_summarization(text)
        return summary

    # Split tokens into chunks
    chunks = []
    for i in range(0, token_count, max_tokens_per_chunk):
        chunk_tokens = tokens[i:i + max_tokens_per_chunk]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    # Summarize each chunk asynchronously
    tasks = [call_openai_summarization(chunk) for chunk in chunks]
    summaries = await asyncio.gather(*tasks)

    # Combine summaries
    combined_summary = ' '.join(filter(None, summaries))
    logger.debug('Text summarization completed.')
    return combined_summary

# Asynchronous function to process documents
async def process_documents(earnings_call_text, industry_report_text, economic_report_text):
    # Summarize the texts
    logger.debug('Summarizing the extracted texts.')
    summaries = await asyncio.gather(
        summarize_text_async(earnings_call_text),
        summarize_text_async(industry_report_text),
        summarize_text_async(economic_report_text)
    )

    if not all(summaries):
        logger.warning('Error summarizing one or more documents.')
        return None

    earnings_call_summary, industry_report_summary, economic_report_summary = summaries

    # Analyze sentiments on the summaries concurrently
    logger.debug('Analyzing sentiments on the summaries.')
    sentiments = await asyncio.gather(
        call_openai_analyze_sentiment(earnings_call_summary, "earnings call transcript"),
        call_openai_analyze_sentiment(industry_report_summary, "industry report"),
        call_openai_analyze_sentiment(economic_report_summary, "economic report")
    )

    return sentiments

# Function to run async code from sync context
def run_async_function(coroutine):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    except RuntimeError as e:
        # Create a new event loop if necessary
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coroutine)
        loop.close()
        return result

# Function to process financial CSV files
def process_financial_csv(file_path, csv_name):
    logger.debug(f'Processing CSV file: {file_path}')
    try:
        # Read the CSV file with proper handling of commas and quotes
        df = pd.read_csv(file_path, index_col=0, thousands=',', quotechar='"')

        # Remove any leading/trailing whitespace from headers and index
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()

        # Remove 'ttm' column if present
        if 'ttm' in df.columns:
            df.drop(columns=['ttm'], inplace=True)

        # Transpose the DataFrame so that dates are in rows
        df = df.transpose()

        # Reset index to make dates a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        # Log the 'Date' column values for debugging
        logger.debug(f"{csv_name} 'Date' column values after processing: {df['Date'].tolist()}")

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

        # Remove rows where 'Date' couldn't be parsed
        df = df[df['Date'].notnull()]

        if df['Date'].isnull().any():
            invalid_dates = df[df['Date'].isnull()]['Date']
            logger.error(f"Invalid dates found in {csv_name} CSV. Invalid entries: {invalid_dates}")
            return None

        # Ensure all other columns are numeric
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaN values with zeros or handle accordingly
        df.fillna(0, inplace=True)

        # Log the columns and data for debugging
        logger.debug(f"Processed {csv_name} CSV columns: {df.columns.tolist()}")
        logger.debug(f"{csv_name} DataFrame after processing:\n{df.head()}")
        logger.debug(f'Completed processing CSV file: {file_path}')
        return df
    except Exception as e:
        logger.error(f"Error processing {csv_name} CSV: {str(e)}")
        return None

# Function to standardize column names
def standardize_columns(df, column_mappings, csv_name):
    logger.debug(f'Standardizing columns for {csv_name}.')
    df_columns = df.columns.tolist()
    new_columns = {}
    for standard_name, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df_columns:
                new_columns[name] = standard_name
                break
    df.rename(columns=new_columns, inplace=True)
    # Log the new columns for debugging
    logger.debug(f'After standardization, columns in {csv_name}: {df.columns.tolist()}')
    return df

def generate_pdf_report(
    financials, ratios, cagr_values, sentiment_results,
    plots, recommendation, intrinsic_value_per_share,
    stock_price, weighted_total_score, weights, factor_scores,
    company_summary, industry_summary, risks_summary,
    company_logo_path=None
):
    logger.debug('Starting PDF report generation.')
    # Set up the document
    pdf_output = io.BytesIO()
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    # Modify existing styles instead of adding new ones
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceAfter = 12

    styles['Heading2'].fontSize = 14
    styles['Heading2'].leading = 18
    styles['Heading2'].spaceAfter = 10

    styles['Normal'].fontSize = 12
    styles['Normal'].leading = 14

    # Modify existing 'BodyText' style
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 12

    # Define a custom style for centered text
    centered_style = ParagraphStyle('Centered', alignment=TA_CENTER, fontSize=12)

    # Title Page
    elements.append(Paragraph("Financial Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Include the company logo if provided
    if company_logo_path:
        logo = Image(company_logo_path, width=200, height=100)  # Adjust width and height as needed
        elements.append(logo)
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Company: {financials.get('company_name', 'N/A')}", centered_style))
    elements.append(Paragraph(f"Report Date: {pd.Timestamp('today').strftime('%Y-%m-%d')}", centered_style))
    elements.append(PageBreak())

    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    summary_text = f"""
    This report provides a comprehensive financial analysis of {financials.get('company_name', 'the company')}. The analysis includes Discounted Cash Flow (DCF), ratio analysis, time series analysis, sentiment analysis from various reports, and data visualizations. The final recommendation based on the weighted factors is: <b>{recommendation}</b>.
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Company Summary
    elements.append(Paragraph("Company Summary", styles['Heading1']))
    elements.append(Paragraph(company_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Industry Summary
    elements.append(Paragraph("Industry Summary", styles['Heading1']))
    elements.append(Paragraph(industry_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Risk Considerations
    elements.append(Paragraph("Risk Considerations", styles['Heading1']))
    elements.append(Paragraph(risks_summary, styles['Normal']))
    elements.append(Spacer(1, 12))


    # Financial Analysis
    elements.append(Paragraph("Financial Analysis", styles['Heading1']))

    # DCF Analysis
    elements.append(Paragraph("Discounted Cash Flow (DCF) Analysis", styles['Heading2']))
    dcf_text = f"""
    - Intrinsic Value per Share: ${intrinsic_value_per_share:.2f}<br/>
    - Current Stock Price: ${stock_price}<br/>
    - Factor 1 Score (DCF Analysis): {factor_scores['factor1_score']}
    """
    elements.append(Paragraph(dcf_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Ratio Analysis
    elements.append(Paragraph("Ratio Analysis", styles['Heading2']))
    ratio_data = [
        ["Ratio", "Value"],
        ["Debt-to-Equity Ratio", f"{ratios['Debt-to-Equity Ratio']:.2f}"],
        ["Current Ratio", f"{ratios['Current Ratio']:.2f}"],
        ["P/E Ratio", f"{ratios['P/E Ratio']:.2f}"],
        ["P/B Ratio", f"{ratios['P/B Ratio']:.2f}"],
        ["Factor 2 Score (Ratio Analysis)", f"{factor_scores['factor2_score']}"],
    ]
    ratio_table = Table(ratio_data, hAlign='LEFT')
    ratio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(ratio_table)
    elements.append(Spacer(1, 12))

    # Time Series Analysis
    elements.append(Paragraph("Time Series Analysis", styles['Heading2']))
    cagr_data = [
        ["Metric", "CAGR"],
        ["Revenue CAGR", f"{cagr_values['revenue_cagr']:.2%}" if cagr_values['revenue_cagr'] is not None else "N/A"],
        ["Net Income CAGR", f"{cagr_values['net_income_cagr']:.2%}" if cagr_values['net_income_cagr'] is not None else "N/A"],
        ["Total Assets CAGR", f"{cagr_values['assets_cagr']:.2%}" if cagr_values['assets_cagr'] is not None else "N/A"],
        ["Total Liabilities CAGR", f"{cagr_values['liabilities_cagr']:.2%}" if cagr_values['liabilities_cagr'] is not None else "N/A"],
        ["Operating Cash Flow CAGR", f"{cagr_values['cashflow_cagr']:.2%}" if cagr_values['cashflow_cagr'] is not None else "N/A"],
        ["Factor 3 Score (Time Series Analysis)", f"{factor_scores['factor3_score']}"],
    ]
    cagr_table = Table(cagr_data, hAlign='LEFT')
    cagr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(cagr_table)
    elements.append(Spacer(1, 12))

    # Sentiment Analysis
    elements.append(Paragraph("Sentiment Analysis", styles['Heading1']))

    for sentiment in sentiment_results:
        elements.append(Paragraph(sentiment['title'], styles['Heading2']))
        sentiment_text = f"""
        - Sentiment Score: {sentiment['score']:.2f} <br/>
        - Explanation: {sentiment['explanation']} <br/>
        - Factor Score: {sentiment['factor_score']}
        """
        elements.append(Paragraph(sentiment_text, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Data Visualizations
    elements.append(Paragraph("Data Visualizations", styles['Heading1']))

    # Include plots
    for plot_title, plot_image in plots.items():
        elements.append(Paragraph(plot_title, styles['Heading2']))
        img = Image(plot_image, width=500, height=200)
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Final Recommendation
    elements.append(Paragraph("Final Recommendation", styles['Heading1']))
    recommendation_text = f"""
    The weighted total score based on the analysis is: {weighted_total_score}.<br/>
    The final recommendation is: <b>{recommendation}</b>.
    """
    elements.append(Paragraph(recommendation_text, styles['Normal']))

    # Build the PDF
    doc.build(elements)
    logger.debug('PDF report generation completed.')

    pdf_output.seek(0)
    return pdf_output

# Serve the homepage
@app.route('/')
def home():
    logger.debug('Rendering homepage.')
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_financials():
    try:
        logger.info('Starting analyze_financials function.')
        # Initialize a dictionary to hold file paths
        file_paths = {}

        # Define expected files and their types
        expected_files = {
            'income_statement': 'csv',
            'balance_sheet': 'csv',
            'cash_flow': 'csv',
            'earnings_call': 'pdf',
            'industry_report': 'pdf',
            'economic_report': 'pdf',
            'ten_k_report': 'pdf',
            'company_logo': 'image'
        }

        # Loop through the expected files
        for field_name, file_type in expected_files.items():
            uploaded_file = request.files.get(field_name)
            if uploaded_file and allowed_file(uploaded_file.filename, ALLOWED_EXTENSIONS[file_type]):
                filename = secure_filename(uploaded_file.filename)
                # Create a unique file path to prevent overwriting
                if field_name == 'company_logo':
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"logo_{filename}")
                else:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{field_name}_{filename}")
                uploaded_file.save(file_path)
                file_paths[field_name] = file_path
                logger.info(f"Received file: {field_name} - {filename}")
            else:
                if field_name == 'company_logo':
                    # Company logo is optional; handle gracefully
                    file_paths[field_name] = None
                    logger.warning('No company logo uploaded or invalid file type.')
                else:
                    error_message = f'Invalid or missing file for {field_name}. Please upload a valid {file_type.upper()} file.'
                    logger.warning(error_message)
                    return jsonify({'error': error_message}), 400

        # Extract text from the 10-K report
        logger.debug('Extracting text from the 10-K report.')
        ten_k_text = extract_text_from_pdf(file_paths['ten_k_report'])

        if not ten_k_text:
            error_message = 'Error extracting text from the 10-K report. Ensure the PDF contains extractable text.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400
        else:
            logger.info('Extracted text from the 10-K report successfully.')

        # Use GPT-3.5 to generate summaries and risk considerations
        company_summary, industry_summary, risks_summary = run_async_function(
            generate_summaries_and_risks(ten_k_text)
        )

        # Check if the summaries were successfully generated
        if not company_summary or not industry_summary or not risks_summary:
            error_message = 'Error generating summaries and risk considerations using GPT-3.5 Turbo.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 500
        else:
            logger.info('Generated company, industry summaries, and risk considerations successfully.')


        # Retrieve the company name
        company_name = request.form.get('company_name', 'N/A')
        financials = {'company_name': company_name}
        logger.info(f"Company Name: {company_name}")

        # Retrieve and validate user input data from the form
        try:
            wacc = float(request.form['wacc']) / 100  # WACC input
            tax_rate = float(request.form['tax_rate']) / 100  # Tax rate input
            growth_rate = float(request.form['growth_rate']) / 100  # Growth rate input
            stock_price = float(request.form['stock_price'])  # Current stock price input
            logger.info(f"Received financial inputs - WACC: {wacc}, Tax Rate: {tax_rate}, Growth Rate: {growth_rate}, Stock Price: {stock_price}")
        except (ValueError, KeyError) as e:
            error_message = 'Please provide valid numerical inputs for WACC, Tax Rate, Growth Rate, and Stock Price.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        # Retrieve and validate industry benchmarks
        try:
            debt_equity_benchmark = float(request.form['debt_equity_benchmark'])
            current_ratio_benchmark = float(request.form['current_ratio_benchmark'])
            pe_benchmark = float(request.form['pe_benchmark'])
            pb_benchmark = float(request.form['pb_benchmark'])
            logger.info(f"Received industry benchmarks - Debt to Equity: {debt_equity_benchmark}, Current Ratio: {current_ratio_benchmark}, P/E Ratio: {pe_benchmark}, P/B Ratio: {pb_benchmark}")
        except (ValueError, KeyError) as e:
            error_message = 'Please provide valid numerical inputs for all industry benchmarks.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        benchmarks = {
            'debt_to_equity': debt_equity_benchmark,
            'current_ratio': current_ratio_benchmark,
            'pe_ratio': pe_benchmark,
            'pb_ratio': pb_benchmark
        }

        # Process CSV files using the updated function
        logger.debug('Processing CSV files.')
        income_df = process_financial_csv(file_paths['income_statement'], 'income_statement')
        balance_df = process_financial_csv(file_paths['balance_sheet'], 'balance_sheet')
        cashflow_df = process_financial_csv(file_paths['cash_flow'], 'cash_flow')

        if income_df is None or balance_df is None or cashflow_df is None:
            error_message = 'Error processing CSV files. Please ensure they are correctly formatted.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        # Standardize column names
        logger.debug('Standardizing CSV column names.')
        income_columns = {
            'Revenue': ['TotalRevenue', 'Revenue', 'Total Revenue', 'Sales'],
            'Net Income': ['NetIncome', 'Net Income', 'Net Profit', 'Profit After Tax'],
        }

        balance_columns = {
            'Total Assets': ['TotalAssets', 'Total Assets'],
            'Total Liabilities': ['TotalLiabilities', 'Total Liabilities', 'TotalLiabilitiesNetMinorityInterest'],
            'Shareholders Equity': ['TotalEquity', 'Shareholders Equity', 'Total Equity', 'StockholdersEquity'],
            'Current Assets': ['CurrentAssets', 'Current Assets'],
            'Current Liabilities': ['CurrentLiabilities', 'Current Liabilities'],
            'CurrentDebt': ['CurrentDebt', 'ShortTermDebt', 'Short-Term Debt', 'Current Debt'],
            'Long-Term Debt': ['LongTermDebt', 'Long-Term Debt', 'Non-Current Debt', 'Long-Term Debt'],
            'Total Shares Outstanding': ['SharesIssued', 'ShareIssued', 'Total Shares Outstanding', 'Total Shares', 'OrdinarySharesNumber'],
            'Inventory': ['Inventory', 'Inventories'],
        }

        cashflow_columns = {
            'Operating Cash Flow': ['OperatingCashFlow', 'Operating Cash Flow', 'Cash from Operations'],
            'Capital Expenditures': ['CapitalExpenditures', 'Capital Expenditures', 'CapEx', 'CapitalExpenditure', 'Capital Expenditure'],
        }

        income_df = standardize_columns(income_df, income_columns, 'income_statement')
        balance_df = standardize_columns(balance_df, balance_columns, 'balance_sheet')
        cashflow_df = standardize_columns(cashflow_df, cashflow_columns, 'cash_flow')

        # Log columns after standardization
        logger.debug(f'Income Statement Columns: {income_df.columns.tolist()}')
        logger.debug(f'Balance Sheet Columns: {balance_df.columns.tolist()}')
        logger.debug(f'Cash Flow Statement Columns after standardization: {cashflow_df.columns.tolist()}')

        # Verify 'Date' column exists in all DataFrames
        for df, name in [(income_df, 'income_statement'), (balance_df, 'balance_sheet'), (cashflow_df, 'cash_flow')]:
            if 'Date' not in df.columns:
                error_message = f"'Date' column missing in {name} CSV after processing. Please ensure the CSV has dates as column headers."
                logger.error(error_message)
                return jsonify({'error': error_message}), 400
            else:
                logger.debug(f"'Date' column found in {name} CSV.")

        # Sort data by date
        income_df.sort_values('Date', inplace=True)
        balance_df.sort_values('Date', inplace=True)
        cashflow_df.sort_values('Date', inplace=True)

        # Reset index after sorting
        income_df.reset_index(drop=True, inplace=True)
        balance_df.reset_index(drop=True, inplace=True)
        cashflow_df.reset_index(drop=True, inplace=True)

        # Ensure numeric data types
        numeric_columns = [
            'Revenue', 'Net Income', 'Total Assets', 'Total Liabilities',
            'Shareholders Equity', 'Current Assets', 'Current Liabilities',
            'CurrentDebt', 'Long-Term Debt', 'Total Shares Outstanding',
            'Operating Cash Flow', 'Capital Expenditures', 'Inventory'
        ]
        for col in numeric_columns:
            if col in income_df.columns:
                income_df[col] = pd.to_numeric(income_df[col], errors='coerce')
            if col in balance_df.columns:
                balance_df[col] = pd.to_numeric(balance_df[col], errors='coerce')
            if col in cashflow_df.columns:
                cashflow_df[col] = pd.to_numeric(cashflow_df[col], errors='coerce')

        # Fill NaN values with zeros
        income_df.fillna(0, inplace=True)
        balance_df.fillna(0, inplace=True)
        cashflow_df.fillna(0, inplace=True)
        logger.info('Processed CSV files successfully.')

        # Extract financial data from DataFrames
        try:
            logger.debug('Extracting financial data from DataFrames.')
            # Balance Sheet data
            latest_date = balance_df['Date'].max()

            # Extract Long-Term Debt
            if 'Long-Term Debt' in balance_df.columns:
                long_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'Long-Term Debt'].values[0]
            elif 'LongTermDebt' in balance_df.columns:
                long_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'LongTermDebt'].values[0]
            else:
                long_term_debt = 0
                logger.warning("Long-Term Debt not found in balance sheet columns.")

            # Extract Short-Term Debt (CurrentDebt)
            if 'CurrentDebt' in balance_df.columns:
                short_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'CurrentDebt'].values[0]
            else:
                short_term_debt = 0
                logger.warning("Current Debt not found in balance sheet columns.")

            total_debt = long_term_debt + short_term_debt

            shareholders_equity = balance_df.loc[balance_df['Date'] == latest_date, 'Shareholders Equity'].values[0]
            current_assets = balance_df.loc[balance_df['Date'] == latest_date, 'Current Assets'].values[0]
            current_liabilities = balance_df.loc[balance_df['Date'] == latest_date, 'Current Liabilities'].values[0]
            total_shares_outstanding = balance_df.loc[balance_df['Date'] == latest_date, 'Total Shares Outstanding'].values[0]
            inventory = balance_df.loc[balance_df['Date'] == latest_date, 'Inventory'].values[0] if 'Inventory' in balance_df.columns else 0
            book_value_per_share = safe_divide(shareholders_equity, total_shares_outstanding)

            # Income Statement data
            net_income = income_df['Net Income'].iloc[-1]
            revenue = income_df['Revenue'].iloc[-1]
            eps = safe_divide(net_income, total_shares_outstanding)

            # Cash Flow Statement data
            # Extract Operating Cash Flow
            if 'Operating Cash Flow' in cashflow_df.columns:
                operating_cash_flow = cashflow_df['Operating Cash Flow']
            else:
                logger.error("Operating Cash Flow column not found in cash flow statement.")
                return jsonify({'error': 'Operating Cash Flow data is missing.'}), 400

            # Extract Capital Expenditures
            if 'Capital Expenditures' in cashflow_df.columns:
                capital_expenditures = cashflow_df['Capital Expenditures']
            else:
                logger.error("Capital Expenditures column not found in cash flow statement.")
                return jsonify({'error': 'Capital Expenditures data is missing.'}), 400

            free_cash_flows = operating_cash_flow - capital_expenditures
            free_cash_flows = free_cash_flows.reset_index(drop=True)

            # DCF analysis enhancements
            if len(free_cash_flows) < 2:
                error_message = 'Not enough data to project future free cash flows.'
                logger.warning(error_message)
                return jsonify({'error': error_message}), 400

            # Calculate historical growth rates
            historical_growth_rates = free_cash_flows.pct_change().dropna()
            average_growth_rate = historical_growth_rates.mean()
            logger.debug(f'Average historical growth rate: {average_growth_rate}')

            # Determine projected growth rate
            projected_growth_rate = min(average_growth_rate, growth_rate)
            logger.debug(f'Projected growth rate used: {projected_growth_rate}')

            # Project future free cash flows
            projection_years = 5
            last_free_cash_flow = free_cash_flows.iloc[-1]
            projected_free_cash_flows = [
                last_free_cash_flow * (1 + projected_growth_rate) ** i
                for i in range(1, projection_years + 1)
            ]
            logger.debug(f'Projected free cash flows: {projected_free_cash_flows}')

            # Ensure WACC > growth rate
            if wacc <= growth_rate:
                error_message = 'WACC must be greater than the growth rate for DCF calculation.'
                logger.warning(error_message)
                return jsonify({'error': error_message}), 400

            # Calculate terminal value
            terminal_value = projected_free_cash_flows[-1] * (1 + growth_rate) / (wacc - growth_rate)
            logger.debug(f'Calculated terminal value: {terminal_value}')

            # Perform DCF analysis
            dcf_value = dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years)
            logger.debug(f'DCF value calculated: {dcf_value}')

            # Calculate intrinsic value per share
            intrinsic_value_per_share = safe_divide(dcf_value, total_shares_outstanding)
            logger.debug(f'Intrinsic value per share: {intrinsic_value_per_share}')

            # Factor 1 Score with Margin of Safety
            if intrinsic_value_per_share is None:
                factor1_score = 0
                logger.warning('Intrinsic value per share could not be calculated.')
            else:
                # Calculate the 10% margin of safety bounds
                upper_bound = stock_price * 1.10  # 10% higher
                lower_bound = stock_price * 0.90  # 10% lower
    
                if intrinsic_value_per_share > upper_bound:
                    factor1_score = 1  # Intrinsic value is more than 10% higher
                elif intrinsic_value_per_share < lower_bound:
                    factor1_score = -1  # Intrinsic value is more than 10% lower
                else:
                    factor1_score = 0  # Intrinsic value is within the ±10% range

            logger.debug(f'Factor 1 Score: {factor1_score} (Intrinsic Value: {intrinsic_value_per_share}, Stock Price: {stock_price})')


            financials = {
                'total_debt': total_debt,
                'shareholders_equity': shareholders_equity,
                'current_assets': current_assets,
                'current_liabilities': current_liabilities,
                'inventory': inventory,
                'market_price': stock_price,
                'eps': eps,
                'book_value_per_share': book_value_per_share,
                'net_income': net_income,
                'revenue': revenue,
                'company_name': request.form.get('company_name', 'N/A')
            }
            logger.info('Extracted financial data successfully.')
        except Exception as e:
            error_message = f'Error extracting financial data: {str(e)}'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        # Perform ratio analysis
        logger.debug('Performing ratio analysis.')
        ratios = calculate_ratios(financials, benchmarks)
        logger.info('Performed ratio analysis.')

        # Prepare data for benchmark comparison plot
        benchmark_comparison = {
            'Ratios': ['Debt-to-Equity Ratio', 'Current Ratio', 'P/E Ratio', 'P/B Ratio'],
            'Company': [
                ratios.get('Debt-to-Equity Ratio', 0) if ratios.get('Debt-to-Equity Ratio') is not None else 0,
                ratios.get('Current Ratio', 0) if ratios.get('Current Ratio') is not None else 0,
                ratios.get('P/E Ratio', 0) if ratios.get('P/E Ratio') is not None else 0,
                ratios.get('P/B Ratio', 0) if ratios.get('P/B Ratio') is not None else 0,
            ],
            'Industry': [
                benchmarks['debt_to_equity'],
                benchmarks['current_ratio'],
                benchmarks['pe_ratio'],
                benchmarks['pb_ratio'],
            ]
        }

        # Factor 2 score from ratios
        factor2_score = ratios['Normalized Factor 2 Score']

        # Extracting text from PDF files for sentiment analysis
        logger.debug('Extracting text from PDF files for sentiment analysis.')
        earnings_call_text = extract_text_from_pdf(file_paths['earnings_call'])
        industry_report_text = extract_text_from_pdf(file_paths['industry_report'])
        economic_report_text = extract_text_from_pdf(file_paths['economic_report'])

        if not all([earnings_call_text, industry_report_text, economic_report_text]):
            error_message = 'Error extracting text from one or more PDF files. Ensure PDFs contain extractable text.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        # Run the summarization and sentiment analysis asynchronously
        sentiments = run_async_function(
            process_documents(earnings_call_text, industry_report_text, economic_report_text)
        )

        if sentiments is None:
            error_message = 'An error occurred during document processing.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        # Unpack the sentiments
        (earnings_call_score, earnings_call_explanation), \
        (industry_report_score, industry_report_explanation), \
        (economic_report_score, economic_report_explanation) = sentiments

        # Map sentiment scores to factor scores
        def map_sentiment_to_score(sentiment_score):
            if sentiment_score is None:
                return 0
            elif sentiment_score > 0.5:
                return 1
            elif sentiment_score < -0.5:
                return -1
            else:
                return 0

        factor4_score = map_sentiment_to_score(earnings_call_score)
        factor5_score = map_sentiment_to_score(industry_report_score)
        factor6_score = map_sentiment_to_score(economic_report_score)

        # Factor 3: Time Series Analysis
        try:
            logger.debug('Performing time series analysis.')
            n_periods = len(income_df) - 1  # Assuming annual data

            if n_periods < 1:
                error_message = 'Not enough data for time series analysis.'
                logger.warning(error_message)
                return jsonify({'error': error_message}), 400

            # Revenue CAGR
            revenue_cagr = calculate_cagr(
                income_df['Revenue'].iloc[0],
                income_df['Revenue'].iloc[-1],
                n_periods
            )

            # Net Income CAGR
            net_income_cagr = calculate_cagr(
                income_df['Net Income'].iloc[0],
                income_df['Net Income'].iloc[-1],
                n_periods
            )

            # Assets CAGR
            assets_cagr = calculate_cagr(
                balance_df['Total Assets'].iloc[0],
                balance_df['Total Assets'].iloc[-1],
                n_periods
            )

            # Liabilities CAGR
            liabilities_cagr = calculate_cagr(
                balance_df['Total Liabilities'].iloc[0],
                balance_df['Total Liabilities'].iloc[-1],
                n_periods
            )

            # Cash Flow CAGR
            cashflow_cagr = calculate_cagr(
                cashflow_df['Operating Cash Flow'].iloc[0],
                cashflow_df['Operating Cash Flow'].iloc[-1],
                n_periods
            )

            # Initialize scores
            factor3_scores = []

            # Revenue Score
            if revenue_cagr is not None:
                revenue_score = 1 if revenue_cagr > 0 else -1
                factor3_scores.append(revenue_score)

            # Net Income Score
            if net_income_cagr is not None:
                net_income_score = 1 if net_income_cagr > 0 else -1
                factor3_scores.append(net_income_score)

            # Assets Score
            if assets_cagr is not None:
                assets_score = 1 if assets_cagr > 0 else -1
                factor3_scores.append(assets_score)

            # Liabilities Score
            if liabilities_cagr is not None:
                liabilities_score = -1 if liabilities_cagr > 0 else 1
                factor3_scores.append(liabilities_score)

            # Cash Flow Score
            if cashflow_cagr is not None:
                cashflow_score = 1 if cashflow_cagr > 0 else -1
                factor3_scores.append(cashflow_score)

            # Calculate Factor 3 Score
            factor3_score = sum(factor3_scores)

            # Normalize the score to -1, 0, or 1
            if factor3_score > 0:
                factor3_score = 1
            elif factor3_score < 0:
                factor3_score = -1
            else:
                factor3_score = 0

            logger.info('Performed time series analysis.')
        except Exception as e:
            error_message = f'Error performing time series analysis: {str(e)}'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        # Generate plots
        logger.debug('Generating plots for data visualization.')
        plots = {}

        # Generate benchmark comparison plot
        benchmark_plot = generate_benchmark_comparison_plot(benchmark_comparison)
        plots['Company vs. Industry Benchmarks'] = benchmark_plot
        logger.info('Benchmark comparison plot generated.')

        # Dates for plotting
        dates = income_df['Date'].dt.strftime('%Y-%m-%d').tolist()

        # Revenue Plot
        plots['Revenue Over Time'] = generate_plot(
            dates,
            income_df['Revenue'].tolist(),
            'Revenue Over Time',
            'Date',
            'Revenue'
        )

        # Net Income Plot
        plots['Net Income Over Time'] = generate_plot(
            dates,
            income_df['Net Income'].tolist(),
            'Net Income Over Time',
            'Date',
            'Net Income'
        )

        # Operating Cash Flow Plot
        cashflow_dates = cashflow_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        plots['Operating Cash Flow Over Time'] = generate_plot(
            cashflow_dates,
            cashflow_df['Operating Cash Flow'].tolist(),
            'Operating Cash Flow Over Time',
            'Date',
            'Operating Cash Flow'
        )

        # Total Assets and Liabilities Plot
        balance_dates = balance_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        plt.figure(figsize=(8, 4))
        plt.plot(balance_dates, balance_df['Total Assets'].tolist(), marker='o', label='Total Assets')
        plt.plot(balance_dates, balance_df['Total Liabilities'].tolist(), marker='o', label='Total Liabilities')
        plt.title('Total Assets and Liabilities Over Time')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='PNG')
        plt.close()
        img_data.seek(0)
        plots['Total Assets and Liabilities Over Time'] = img_data
        logger.debug('Plots generated.')

        # Assign weights to factors
        weights = {
            'factor1': 1,  # DCF Analysis
            'factor2': 1,  # Ratio Analysis
            'factor3': 1,  # Time Series Analysis
            'factor4': 1,  # Earnings Call Sentiment
            'factor5': 1,  # Industry Report Sentiment
            'factor6': 1   # Economic Report Sentiment
        }

        # Calculate weighted total score
        weighted_total_score = (
            factor1_score * weights['factor1'] +
            factor2_score * weights['factor2'] +
            factor3_score * weights['factor3'] +
            factor4_score * weights['factor4'] +
            factor5_score * weights['factor5'] +
            factor6_score * weights['factor6']
        )
        logger.debug(f'Weighted total score calculated: {weighted_total_score}')

        # Determine recommendation based on weighted total score
        if weighted_total_score >= 3:
            recommendation = "Buy"
        elif weighted_total_score <= -3:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        logger.info(f'Final recommendation: {recommendation}')

        # Prepare sentiment_results list
        sentiment_results = [
            {
                'title': 'Earnings Call Sentiment',
                'score': earnings_call_score,
                'explanation': earnings_call_explanation,
                'factor_score': factor4_score
            },
            {
                'title': 'Industry Report Sentiment',
                'score': industry_report_score,
                'explanation': industry_report_explanation,
                'factor_score': factor5_score
            },
            {
                'title': 'Economic Report Sentiment',
                'score': economic_report_score,
                'explanation': economic_report_explanation,
                'factor_score': factor6_score
            },
        ]

        factor_scores = {
            'factor1_score': factor1_score,
            'factor2_score': factor2_score,
            'factor3_score': factor3_score,
            'factor4_score': factor4_score,
            'factor5_score': factor5_score,
            'factor6_score': factor6_score,
        }

        # Generate the PDF report
        logger.debug('Generating PDF report.')
        pdf_output = generate_pdf_report(
            financials=financials,
            ratios=ratios,
            cagr_values={
                'revenue_cagr': revenue_cagr,
                'net_income_cagr': net_income_cagr,
                'assets_cagr': assets_cagr,
                'liabilities_cagr': liabilities_cagr,
                'cashflow_cagr': cashflow_cagr
            },
            sentiment_results=sentiment_results,
            plots=plots,
            recommendation=recommendation,
            intrinsic_value_per_share=intrinsic_value_per_share,
            stock_price=stock_price,
            weighted_total_score=weighted_total_score,
            weights=weights,
            factor_scores=factor_scores,
            company_summary=company_summary,           # Added this line
            industry_summary=industry_summary,         # Added this line
            risks_summary=risks_summary,               # Added this line
            company_logo_path=file_paths.get('company_logo')
        )

        logger.info('Completed analyze_financials function successfully.')
        # Return the PDF file for download
        return send_file(
            pdf_output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='financial_report.pdf'
        )

    except Exception as e:
        logger.exception('An unexpected error occurred during analysis.')
        return jsonify({'error': 'An unexpected error occurred.'}), 500

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("your_script_name:asgi_app", host='0.0.0.0', port=5000)
