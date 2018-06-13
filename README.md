# sentiment_api
Implements a Bidirectional LSTM that utilises word and character embeddings to classify sentiment as positive or negative. Exposes a trained model via a REST API using Django. 

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
- [x] Simple Boostrap interface
- [x] Basic Class-based API endpoint
- [x] Ajax call to API endpoint
- [x] BaseModel class
- [x] Bidirectional LSTM with word and character embeddings
- [x] Vocabulary builder
- [x] DocIdTransformer
- [ ] Model training and storage
- [ ] Local loading and prediction through API endpoint
- [ ] Downloading script for short-text sentiment data
- [ ] Dockerize project
- [ ] Deploy on server

