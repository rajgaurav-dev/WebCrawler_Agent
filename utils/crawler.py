import os
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import numpy as np
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
# 5. Generate Embedding
# -----------------------------
def generate_embedding(text):
    embedding = client.feature_extraction(text)
    return np.array(embedding)


# -----------------------------
# 6. Cosine Similarity
# -----------------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


# -----------------------------
# 7. Rank Links by Semantic Similarity
# -----------------------------
def rank_links(links, target_text, top_k=10):

    print("Generating target embedding...")
    target_embedding = generate_embedding(target_text)

    scored_links = []

    print("Scoring links...")
    for text, url in links:
        combined_text = f"{text} {url}"  # better semantic context
        link_embedding = generate_embedding(combined_text)

        score = cosine_similarity(target_embedding, link_embedding)
        scored_links.append((text, url, score))

    scored_links.sort(key=lambda x: x[2], reverse=True)

    return scored_links[:top_k]


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    homepage = "https://www.deere.com/"

    print("Extracting links...")
    links = extract_links(homepage)
    print(f"Total links found: {len(links)}")

    print("Applying basic filter...")
    links = basic_filter_links(links, homepage)
    print(f"Links after filtering: {len(links)}")

    # Define search intent
    TARGET_DESCRIPTION = "agricultural tractor implements and farm equipment product listings and documentation"

    print("\nRanking links using embeddings...\n")
    top_links = rank_links(links, TARGET_DESCRIPTION, top_k=5)

    print("\nTop Relevant Links:\n")
    for text, url, score in top_links:
        print(f"Score: {score:.4f}")
        print(f"Text: {text}")
        print(f"URL: {url}")
        print("-" * 60)