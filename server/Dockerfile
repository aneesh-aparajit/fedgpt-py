FROM python:3.9
# RUN apk add gcc g++ linux-headers
WORKDIR /usr/server/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir models/
RUN cd models/
COPY "../models/*.py" /usr/server/models/
CMD [ "sh", "run-server.sh" ]
