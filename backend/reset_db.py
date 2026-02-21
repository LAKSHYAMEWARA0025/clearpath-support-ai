import os
import shutil
from config import CHROMA_DB_DIR

if os.path.exists(CHROMA_DB_DIR):
    shutil.rmtree(CHROMA_DB_DIR)
    print(f"Successfully deleted the old database at: {CHROMA_DB_DIR}")
else:
    print(f"No database found at {CHROMA_DB_DIR}. You are good to go!")