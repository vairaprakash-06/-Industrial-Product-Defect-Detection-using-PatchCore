"""Root Streamlit Web Application entry point for Industrial Product Defect Detection.

Launch via:
    streamlit run app.py
or
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run the primary Streamlit application from app/app.py
from app.app import main

if __name__ == "__main__":
    main()
