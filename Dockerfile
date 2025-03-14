# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
# Uncomment the line below if you have a requirements.txt file
# RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the application
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
