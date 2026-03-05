# MediAI - Disease Detection & Prescription System

MediAI is an industry-level disease detection system that leverages a Machine Learning model to diagnose diseases based on user-inputted symptoms and provides suitable medicine prescriptions. 

The application utilizes a **Next.js** frontend with a premium dark medical UI and a **FastAPI** backend supporting a Random Forest disease classification model.

---

## 🚀 Features
- **Symptom-based Diagnosis:** Select symptoms from a comprehensive database to receive AI-predicted diagnoses with confidence scores.
- **Detailed Disease Encyclopedia:** Browse severity, affected body systems, descriptions, precautions, and recommended lifestyle changes for 40+ conditions.
- **Smart Prescriptions:** View tailored medicine recommendations with dosage and notes for each predicted condition.
- **Secure Authentication:** Integrated OAuth (Google, GitHub) & Credentials login via NextAuth.
- **Beautiful UI/UX:** Responsive, fully dark/light-mode togglable interface with premium aesthetics.

---

## 🛠️ Tech Stack
- **Frontend:** Next.js 15, React, Vanilla CSS Modules, NextAuth.js
- **Backend:** FastAPI, Python, Uvicorn
- **Machine Learning:** Scikit-Learn (Random Forest Classifier), Joblib

---

## 💻 Local Setup & Installation

### Prerequisites
- Node.js (v18+ recommended)
- Python (v3.9+ recommended)

### 1. Clone the repository
```bash
git clone https://github.com/swapneelmukherjee2004-bit/MediAI.git
cd MediAI
```

### 2. Backend Setup
Create and activate a Python virtual environment, then install dependencies:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
*(Note: If the pre-trained `model.pkl` is missing, you can regenerate it by running `python model/train.py` from within the backend directory).*

### 3. Frontend Setup
Install NPM dependencies:
```bash
cd ../frontend
npm install
```

### 4. Environment Variables
Create a `.env.local` file inside the `frontend/` directory with the following variables:
```env
# NextAuth required variables
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your_super_secret_string_here

# OAuth Providers (Optional but required for Social Login)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

### 5. Start the Application
You can easily start both the backend API and frontend dev server simultaneously from the project root using the provided bash script:
```bash
# From the root /MediAI directory
./start.sh
```
The application will be accessible at:
- **Frontend:** `http://localhost:3000`
- **Backend API Docs:** `http://localhost:8000/docs`

---

## 📸 Screenshots
*(Add screenshots of your application here)*

---

## 📄 License
This project is licensed under the MIT License.
