# Fingerprint_BGD_App
Blood group detection using fingerprints

-----------------------------------------
Steps to run the solution without Docker:
-----------------------------------------

1.Download and install Python
Download Python 3.10 or 3.9:
Go to the Python Downloads Page.
https://www.python.org/downloads/

Pre-requisites: Download a version of Python from 3.7 to 3.11 (Solution is built with Python 3.10).

1. Create virtual environment(only once)
python -m venv myenv  # Replace with the path to the compatible Python version

2. Activate virtual environment
.\myenv\Scripts\activate

3. Install dependencies to virtual environment (only for the first time or if there are new dependencies added)
pip install -r requirements.txt

4. Run the python app
python application.py

link to access the application after run
http://localhost:5000


---------------------------
Run with Docker
---------------------------

1. Build the Docker image (include the . at the end)
docker build -t my-fbgd-app .

2. Run the Docker container
docker run --name My-FingerPrint-BloodGroupDetection -p 5000:5000 my-fbgd-app

link to access the application after run
http://127.0.0.1:5000

