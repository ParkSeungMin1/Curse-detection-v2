FROM python:3.10.4

RUN mkdir src

COPY . ./src


RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["/bin/bash"]

