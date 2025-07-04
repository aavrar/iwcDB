import ssl
import certifi
import urllib.request

ctx = ssl.create_default_context(cafile=certifi.where())
response = urllib.request.urlopen("https://twitter.com", context=ctx)
print("SSL test response status:", response.status)