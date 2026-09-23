import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "embedding" / "chunks.pkl"


with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


print(f"总共有 {len(chunks)} 个 chunks")


for chunk_id, text in enumerate(chunks):
    print("\n")
    print("=" * 80)

    print("chunk_id:", chunk_id)

    print("-" * 80)

    print(text)
