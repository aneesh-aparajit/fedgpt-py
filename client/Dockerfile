FROM python:3.9-alpine3.17
WORKDIR /usr/client/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "sh", "run-client.sh" ]
