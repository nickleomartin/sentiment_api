from django.shortcuts import render
from django.views.generic.base import View 
from django.http.response import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json


## Auxillary functions
def load_model():
	print("model loaded")
	model = None
	return model


class PredictSentimentView(View):
	""" API endpoint to predict sentiment of text """

	def get(self, request, *args, **kwargs):
		""" GET requests returns a warning message"""
		return JsonResponse({"response":"Use the post method","status":400 })

	def post(self, request, *args, **kwargs):
		""" POST requests take in text and return sentiment"""
		# text = self.request.GET['data']
		json_object = json.loads(request.body.decode("utf-8"))
		text = json_object['input']
		print(text)
		return JsonResponse({"text": text, "sentiment_score": 0.9, "response": "Successful", "status": 200})

	@method_decorator(csrf_exempt)
	def dispatch(self, request, *args, **kwargs):
		""" Handle CSRF """
		return View.dispatch(self, request, *args, **kwargs)
