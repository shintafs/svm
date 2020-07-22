import tweepy
# ACCESS_TOKEN="610128492-Mi55nNDVh5YS6gawj0Vat94kHcIcBSpZRAqe1tBt"
# ACCESS_TOKEN_SECRET="QmprTBUQwjs40xD2vb8egjMjgBJbgEGIQ26jLSA522QxF"
# CONSUMER_KEY="B9oXSo836WBX6xN0rdv1JYy23"
# CONSUMER_SECRET="HZK2lHQQrdYI5c888WNjxOnEcAh4XsIBesqYoR6HjZTOAVVVvI"
ACCESS_TOKEN="610128492-kysToikwTlii1xhWwaGBD4zEG3QkzZWpAGWAeIZ8"
ACCESS_TOKEN_SECRET="VBGv7KVzJYzGteRP7i4jFbtfbeMYzLSNGVuNdDSYUYlDs"
CONSUMER_KEY="LyTY0KP0PBJNp8vzJyuClmuCi"
CONSUMER_SECRET="FOLVgsjyVhuHRMSwWnn6g83JXMYiMJUSizpoxm0oQiYhlQE3by"
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# api = tweepy.API(auth)
# data = api.rate_limit_status()
# print (data['resources']['statuses']['/statuses/home_timeline'])
# print (data['resources']['users']['/users/lookup'])

# print(data)