FROM python:3.12-bullseye

EXPOSE 8000
WORKDIR /app
COPY . .

RUN ./install-deps.sh

CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "8000", "server:app"]
