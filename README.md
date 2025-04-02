# Fingerprint_BGD_App
Blood group detection using fingerprints

--------------------------
Steps to run the solution:
--------------------------

1.Download and install Python
Download Python 3.10 or 3.9:
Go to the Python Downloads Page.
https://www.python.org/downloads/

Download a version of Python from 3.7 to 3.11 (Solution is built with Python 3.10).

2. Install dependencies
pip install flask tensorflow pillow werkzeug lime

3. Create virtual environment
python -m venv myenv  # Replace with the path to the compatible Python version

4. Activate virtual environment
.\myenv\Scripts\activate

5. Install dependencies to virtual environment
pip install flask tensorflow pillow werkzeug  lime

6. Run the python app
python application.py

link to access the application after run
http://127.0.0.1:5000


---------------------------
Run with Docker
---------------------------

# Build the Docker image
docker build -t my-fbgd-app .

# Run the Docker container
docker run --name My-FingerPrint-BloodGroupDetection -p 5000:5000 my-fbgd-app

