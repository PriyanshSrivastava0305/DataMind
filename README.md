# AI Assistant for PDF Question Answering

This AI project allows users to ask questions based on PDF content, supported by web search results, making it an ideal tool for research and documentation analysis.

---


## Overview
This project extracts content from a PDF file and combines it with web search results to answer questions. It’s designed for running in Google Colab, utilizing GPU capabilities to improve performance, particularly when creating a vector store for the PDF content.

---

## Requirements

- **Google Colab** (recommended) or a local Python environment
- **GROQ API key**: For web search access, create an API key at [GROQ API Keys](https://console.groq.com/keys).

---

## Installation

### Using Google Colab

1. **Set Up Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload the `.ipynb` notebook file to your Colab environment.

2. **Select GPU Runtime**:
   - Go to **Runtime > Change runtime type** and select **GPU (T4)** for improved performance.

3. **Install Dependencies**:
   - Run the first cell in the notebook to install dependencies, including `PyPDF2` and `Requests`.

### Running Locally

1. **Clone the Repository**:
   - Clone or download the project files to your local machine.

2. **Install Required Packages:**:
   ```bash
   pip install -r requirements.txt

3. **Run the Application**:
   ```bash
   python main.py

---

## Usage Instructions

1. **Upload PDF File**:

 - In Colab, upload the PDF in the file explorer. For local, ensure the PDF file path is accessible.

2. **Enter API Key**:

 - When prompted, enter your GROQ API key.

3. **Load PDF**:

 - Provide the file path to the PDF, e.g., /content/yourfile.pdf in Colab or ./yourfile.pdf locally.

4. **Ask Questions**:

 - Enter questions, and the model will return answers based on the PDF content and web results.

5. **Exit the Session**:

 - Type 'exit' to end the session.

---


## Example Interaction
 
   ``bash
   Enter your question (type 'exit' to stop): What is the report about?
   Response: The report is about the Human Development Report 2020, which documents...
