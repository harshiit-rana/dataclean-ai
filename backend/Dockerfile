FROM python:3.12-slim-bookworm
# Pinned to Debian 12 (bookworm) rather than the floating "slim" tag -- that tag moved to
# Debian 13 (trixie), which renamed libgdk-pixbuf2.0-0 and broke the apt-get layer below.

# WeasyPrint (PDF audit report export, §8.3) needs Pango/Cairo/GDK-Pixbuf at the OS level --
# pip alone can't provide these, hence this apt-get layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
