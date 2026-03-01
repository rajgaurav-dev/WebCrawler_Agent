# WebCrawler_Agent

WebCrawler_Agent is an AI-powered automation agent that crawls websites to extract structured data such as product dimensions and specifications, then automatically populates Excel sheets. Built in Python, it enables intelligent data extraction, cleaning, and formatting for business reporting, catalog management, and analytics workflows.

---

## 🚀 Features

- 🌐 Website crawling and data extraction  
- 📐 Extract product dimensions and specifications  
- 📊 Automatic Excel sheet population  
- 🧹 Data cleaning and formatting  
- 🤖 Designed for Agentic AI workflows  
- 🔄 Modular and extensible architecture  

---

## 🛠 Tech Stack

- Python 3.11+
- uv (package manager)
- Requests / BeautifulSoup / Playwright
- Pandas
- OpenPyXL
- LangChain / CrewAI (optional for agentic workflows)

---

## 📦 Installation Guide

### 1️⃣ Install `uv` (if not installed)

```bash
pip install uv
```

Or (Windows PowerShell recommended):

```bash
irm https://astral.sh/uv/install.ps1 | iex
```

---

### 2️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd WebCrawler_Agent
```

---

### 3️⃣ Create Virtual Environment

```bash
uv venv
```

Activate:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
uv sync
```

This installs all required libraries using:
- `pyproject.toml`
- `uv.lock`

---

## ▶️ Running the Project

```bash
uv run python main.py
```

---

## 📁 Project Structure (Planned)

```
WebCrawler_Agent/
│
├── main.py
├── crawler/
├── agents/
├── utils/
├── output/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🔄 Project Status

An installation guide has been added to the repository.  
Documentation and features will continue to evolve as the project progresses.

---

## 🎯 Future Enhancements

- Intelligent dimension detection using LLM
- Multi-site crawling support
- Automated reporting dashboard
- RAG integration for product queries
- Cloud deployment support

---

## 📜 License

MIT License (or update as needed)

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.