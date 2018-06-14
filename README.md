# sentiment_api
Implements a Bidirectional LSTM that utilises word (and character embeddings later) to classify sentiment as positive or negative. Exposes a trained model via a REST API using Django. 

Getting Setup
-------------
Clone the repo:
```
git clone git@github.com:NickLeoMartin/sentiment_api.git
```

Head into the repo and create and activate a virtual environment:
```
virtualenv --no-site-packages -p python3 venv
source venv/bin/activate
```

Install the packages used:
```
pip install -r requirements.txt
```

Running The Repo Locally:
-------------------------
Access the frontend locally:
```
./manage.py runserver 
```

To-Do
-----
V1:
- [x] Simple Boostrap interface
- [x] Basic Class-based API endpoint
- [x] Ajax call to API endpoint
- [x] BaseModel class
- [x] Bidirectional LSTM with word
- [x] Vocabulary builder
- [x] DocIdTransformer
- [x] Model training
- [x] Model storage
- [x] Model wrapper
- [x] Model prediction 
- [x] Local loading and prediction through API endpoint
- [x] Downloading of weights from git-lfs
- [ ] Dockerize project
- [ ] Deploy on server

V2:
- [ ] Hyperparameter tuning, training summary statistic, more comprehensive dataset 
- [ ] Asynchronous prediction with Celery
- [ ] Downloading script for short-text sentiment data
- [ ] Character embeddings
- [ ] Multi-task learning for entities and sentiment
