import os
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import json
from huggingface_hub import InferenceClient


# -----------------------------
# 1. Load Environment Variables
# -----------------------------
load_dotenv()

HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN")

if not HUGGINGFACE_KEY:
    raise ValueError("HUGGINGFACE_HUB_ACCESS_TOKEN not found in environment variables.")


# -----------------------------
# 2. Initialize HuggingFace Embedding Client
# -----------------------------
client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HUGGINGFACE_KEY
)


# -----------------------------
# 3. Extract Links
# -----------------------------
def extract_links(url: str):
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(url, a_tag["href"])
        link_text = a_tag.get_text(strip=True).lower()

        if link_text:
            links.append((link_text, full_url))

    return links


# -----------------------------
# 4. Basic Filtering (Same Domain Only)
# -----------------------------
def basic_filter_links(links, base_url):
    filtered = []
    base_domain = urlparse(base_url).netloc

    for text, url in links:
        parsed = urlparse(url)

        if parsed.netloc != base_domain:
            continue

        if "#" in url:
            continue

        filtered.append((text, url))

    return filtered


# -----------------------------
# 5. Generate Embeddings
# -----------------------------
def generate_embeddings(text):
    embedding = client.feature_extraction(text)
    return embedding


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    homepage = "https://www.tgschmeiser.com/"

    print("Extracting links...")
    links = extract_links(homepage)
    print(f"Total links found: {len(links)}")

    print("Applying basic filter...")
    links = basic_filter_links(links, homepage)
    print(f"Links after filtering: {len(links)}")

    print("\nGenerating embeddings for first 5 links...\n")

    for text, url in links[:5]:  # Limit for demo
        embedding = generate_embeddings(text)

        print(f"Text: {text}")
        print(f"URL: {url}")
        print(f"Embedding length: {len(embedding)}")
        print("-" * 50)