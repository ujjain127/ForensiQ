"""
ForensiQ Application Launcher
============================
Starts the professional ForensiQ web application
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting ForensiQ Professional Interface...")
    print("=" * 50)
    
    # Change to the correct directory
    os.chdir(r"c:\Users\Admin\ForensiQ")
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8502",
            "--server.headless", "true",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ ForensiQ application stopped")

if __name__ == "__main__":
    main()
