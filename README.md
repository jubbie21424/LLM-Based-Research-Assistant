# LLM-Based Research Assistant

## Overview
This is an individual assignment for MSIN0231 Machine Learning for Business. The project implements a web-based research assistant using Large Language Models (LLMs) to help users with research tasks. It leverages Google Gemini 2.5 Flash for natural language processing and Wikipedia as a knowledge source for retrieving relevant information.

## Features
- **Interactive Web Interface**: Built with Streamlit for an easy-to-use web application.
- **LLM Integration**: Uses Google Gemini 2.5 Flash for generating responses and insights.
- **Wikipedia Retrieval**: Automatically fetches and summarizes relevant Wikipedia articles.
- **Source Citation**: Provides clean, formatted sources with excerpts for transparency.
- **Text Cleaning**: Handles encoding issues and formats text for better readability.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd LLM-Based-Research-Assistant
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Obtain a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. In the sidebar:
   - Select the LLM (currently only Gemini 2.5 Flash is supported).
   - Enter your Gemini API Key.

4. Use the main interface to input your research queries. The assistant will retrieve relevant Wikipedia information and generate responses using the LLM.

## Dependencies
- streamlit
- langchain
- langchain-community
- langchain-google-genai
- langchain-core
- wikipedia
- google-generativeai

## Project Structure
- `app.py`: Main Streamlit application code.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## License
No license required
