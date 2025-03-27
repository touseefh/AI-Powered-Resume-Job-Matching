# HireSync.ai - AI-Powered Resume & Job Matching

## Overview
HireSync.ai is an AI-powered resume and job matching platform that leverages **FAISS** for semantic search, **Google Gemini AI** for AI-driven feedback, and **Streamlit** for an interactive web-based UI. This tool allows HR professionals to create job descriptions, and job seekers to upload resumes to find the best job matches. AI feedback is also generated to help candidates improve their resumes.

## Features
- **Job Posting by HR**: HR professionals can add job descriptions, specify required skills, experience, location, qualifications, etc.
- **Resume Upload**: Candidates can upload resumes in PDF format.
- **FAISS-Powered Matching**: Resumes are matched with job descriptions using **FAISS** and **Sentence-BERT** embeddings.
- **AI Feedback on Resumes**: Google Gemini AI analyzes resumes and provides suggestions on missing skills, improvements, and match percentage.
- **Streamlit UI**: A user-friendly web interface for job creation and resume submission.


## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository
```sh
git clone https://github.com/touseefh/AI-Powered-Resume-Job-Matching.git
cd AI-Powered-Resume-Job-Matching
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Configure API Key for Google Gemini AI
You need a **Google Gemini API key** for AI feedback generation. Replace the placeholder API key in the code with your actual key.
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY_HERE")
```

### Step 4: Run the Application
```sh
streamlit run app.py
```

## Usage

### HR: Creating Job Descriptions
1. Open the web app in your browser.
2. Use the sidebar to add a new job description.
3. Enter job title, required skills, location, German proficiency, availability, visa requirements, experience, qualification level, industry, job type, and a detailed job description.
4. Click **"Add Job"** to save the job listing.

### Candidates: Uploading Resumes
1. Click on the **"Upload Resume"** button and select a PDF file.
2. The system extracts text from the resume and matches it with job listings.
3. If a match is found, the best-matched job description is displayed.
4. AI-generated feedback is provided on how to improve the resume.

## Technology Stack
- **FAISS**: Efficient similarity search for resume matching.
- **Sentence-BERT**: Text embeddings for job and resume comparison.
- **Google Gemini AI**: AI feedback on resume quality.
- **Streamlit**: Interactive web UI.
- **PyPDF2**: PDF text extraction.
- **Matplotlib & Seaborn**: Visualization (future enhancements).



## Future Enhancements
- **Advanced Resume Analysis**: Adding sentiment analysis, keyword extraction, and structured skill categorization.
- **More Job Filters**: Industry-specific weightage, salary range, remote/hybrid jobs.
- **Dashboard for HR**: Visual analytics of job postings and applicant tracking.
- **Database Integration**: Use a database instead of in-memory storage.
---

## Author
Developed by **HireSync.ai** 🚀

