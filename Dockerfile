# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /application

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Start the app
CMD ["python", "application.py"]