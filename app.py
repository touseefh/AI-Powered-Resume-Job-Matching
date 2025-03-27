import faiss
import numpy as np
import PyPDF2
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Google Gemini AI
genai.configure(api_key="Enter Your API Key")

# Load Sentence Transformer Model
embedder = SentenceTransformer("bert-base-nli-mean-tokens")

# Store job descriptions manually entered by HR
job_list = []

# Function to extract text from uploaded PDF resumes
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted.strip() + "\n"
    return text.lower()

# Function to create FAISS index
def create_faiss_index():
    if not job_list:
        return None
    job_descriptions = [job["full_description"] for job in job_list]
    job_embeddings = np.array([embedder.encode(desc) for desc in job_descriptions], dtype="float32")
    index = faiss.IndexFlatIP(job_embeddings.shape[1])  # Inner Product for Cosine Similarity
    index.add(job_embeddings)
    return index

# Function to find best job match
def find_best_match(resume_text, index):
    if index is None or len(job_list) == 0:
        return None  # Return None when no job descriptions are available.
    resume_embedding = np.array([embedder.encode(resume_text)], dtype="float32")
    _, best_match_idx = index.search(resume_embedding, 1)
    
    # Ensure you return a valid job dictionary if a match is found
    if best_match_idx[0][0] != -1:
        best_match_job = job_list[best_match_idx[0][0]]
        return best_match_job, best_match_idx[0][0]
    else:
        return None  # If no valid match found

# Function to generate AI feedback
def generate_feedback(resume_text, job_description):
    prompt = f"""
    Job Description: {job_description}
    Candidate's Resume: {resume_text}
    
    Analyze the resume based on the job description and provide:
    - A match percentage (0-100%).
    - Key skills missing in the resume.
    - Suggested improvements to increase the match.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    return response.text

# Streamlit UI
st.set_page_config(page_title="HireSync.ai", page_icon="📝")
st.title("📝 HireSync.ai - AI-Powered Resume & Job Matching")

# Admin Section - Job Creation
with st.sidebar:
    st.header("🔧 HR: Create Job Descriptions")
    job_title = st.text_input("Job Title")
    job_skills = st.text_area("Required Skills (comma-separated)")
    job_location = st.text_input("Preferred Location")
    job_german_level = st.selectbox("German Proficiency", ["Any", "A1", "A2", "B1", "B2"])
    job_availability = st.selectbox("Joining Availability", ["Immediate", "1 month", "2 months"])
    job_visa_required = st.selectbox("Visa Requirement", ["Yes", "No"])
    job_experience = st.slider("Required Experience (Years)", 0, 20, 3)  # New filter
    job_qualification = st.selectbox("Qualification Level", ["Any", "Bachelor's", "Master's", "PhD"])  # New filter
    job_industry = st.text_input("Industry")  # New filter
    job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Internship"])  # New filter
    job_description = st.text_area("Detailed Job Description")

    if st.button("Add Job"):
        full_description = f"""
        {job_title}  
        Skills: {job_skills}.  
        Location: {job_location}.  
        German Level: {job_german_level}.  
        Availability: {job_availability}.  
        Visa Required: {job_visa_required}.  
        Experience: {job_experience} years.  
        Qualification: {job_qualification}.  
        Industry: {job_industry}.  
        Job Type: {job_type}.
        
        Job Description: {job_description}
        """
        job_list.append({
            "title": job_title,
            "skills": job_skills.split(","),
            "location": job_location,
            "german_level": job_german_level,
            "availability": job_availability,
            "visa": job_visa_required,
            "experience": job_experience,
            "qualification": job_qualification,
            "industry": job_industry,
            "job_type": job_type,
            "full_description": full_description
        })
        st.success(f"Added: {job_title}")

# Create FAISS index
faiss_index = create_faiss_index()

# Resume Upload Section
uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Find the best job match
    best_match = find_best_match(resume_text, faiss_index)
    
    if best_match is not None:
        best_match_job, best_match_idx = best_match

        # Generate AI feedback
        feedback = generate_feedback(resume_text, best_match_job["full_description"])

        # Display results
        st.subheader("🔍 Best Matched Job Description:")
        st.write(best_match_job["full_description"])

        st.subheader("📊 AI Feedback & Suggestions:")
        st.write(feedback)
        
    else:
        st.error("No matching job found.")
