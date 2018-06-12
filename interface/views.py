from django.shortcuts import render
from django.views.generic.base import View 




class IndexView(View):
	template_name = "interface/01_index.html"

	def get(self, request):
		""" Returns input form """
		context = {"form": "!@#$%&"}
		return render(request, self.template_name,context)






















