FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


# Start the container
CMD ["python3", "-u", "rp_handler.py"]