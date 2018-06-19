from django.shortcuts import render
from django.views.generic.base import View 
from django.http.response import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import redis
import uuid
import tensorflow as tf

from sentiment_api import settings
from models.wrapper import SentimentAnalysisModel

## TODO: SORT OUT ERROR HANDLING LOGIC

## Get redis connection
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

class PredictSentimentView(View):
	""" API endpoint to predict sentiment of text """

	def get(self, request, *args, **kwargs):
		""" GET requests returns a warning message"""
		return JsonResponse({"response":"Use the post method","status": 400})

	def post(self, request, *args, **kwargs):
		""" POST requests take in text and return sentiment"""
		data = {"success": False, "text":""}

		try:
			json_object = json.loads(request.body.decode("utf-8"))
			text = json_object['input']
			data["text"] = text
		except:
			return JsonResponse({"response": "json object does not contain 'input'", "status": 400})

		if text!="":

			## Generate UUID for request
			request_id = str(uuid.uuid4())
			payload = {"id": request_id, "text": text}

			## Add to Redis queue
			db.rpush(settings.REDIS_REQUEST_QUEUE, json.dumps(payload))

			while True:

				## Poll for id of request text
				output = db.get(request_id)

				if output is not None:
					## Load dict with json
					output = json.loads(output.decode('utf-8'))

					## Set context dict
					data["sentiment_score"] = output["sentiment_score"]
					data["success"] = True
					data["status"] = 200

					## Delete instance
					db.delete(request_id)

					## Return context object
					return JsonResponse(data)
		else:
			return JsonResponse({"text": "Please enter a comment", "sentiment_score": None, "response": "Successful", "status": 200})

	@method_decorator(csrf_exempt)
	def dispatch(self, request, *args, **kwargs):
		""" Handle Cross-Site Request Forgery """
		return View.dispatch(self, request, *args, **kwargs)



