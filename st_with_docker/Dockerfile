FROM python:3.8-slim

# Meta-data

LABEL maintainer = 'Abhishek Yadav <abhishek.aiem@gmail.com>' \
    description = 'Docker Image Classification using streamlit\
    This image contains a pkl file of predictive model \
    Fitting and pickling was done at another point. This \
    is the FastAI chapter 2 app'

# Setting the working dir to /app
WORKDIR /app

# copy the current directory contents into the container at /app
# COPY ../export.pkl /app
COPY . /app

# pip install  --no-cache-dir 
RUN pip --no-cache-dir install -r /app/requirements.txt

# Make a port available to the outside of this container
EXPOSE 8501

# ENTRYPOINT allows us to specify the default executable
# ENTRYPOINT ["streamlit", "run"]

# CMD sets default arguments to executable which may be overwritten when using docker
# CMD ["app.py"]

CMD streamlit run app.py --server.port $PORT