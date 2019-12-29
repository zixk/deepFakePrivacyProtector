from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    return HttpResponse("Hello world. this is the obfus index.");

def obfuscations(request, participant_num):
    return HttpResponse("You are looking at obfuscations for participant %s." %participant_num)