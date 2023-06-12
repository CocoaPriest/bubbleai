FROM python:3
WORKDIR /server/api
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000