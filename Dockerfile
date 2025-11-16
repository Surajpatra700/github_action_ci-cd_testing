FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8099
CMD ["uvicorn", "semantic_search:app", "--host", "0.0.0.0", "--port", "8099", "--reload"]