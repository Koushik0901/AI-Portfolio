#import all the libraries
import discord
from discord.ext import commands
from tensorflow import keras
import json
import numpy as np
import random
import pickle
#open the intents file
with open("intents.json") as file:
    data = json.load(file)

#load the trained model
model = keras.models.load_model('./chat_model.h5')

# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

#create discord client
class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')
    
    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return
       
        else:
            inps = message.content #getting he input
            inp = inps.lower() #convert to lower case
            if inp[0] == '!': #if the input starts with '!' then run the predictions on it
                result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp[1:]]),
                                                truncating='post', maxlen=20))[0]
                tag = lbl_encoder.inverse_transform([np.argmax(result)])#getting the tags
            
                if np.argmax(result) > 0.60:#setting a threshold
                    for i in data['intents']:
                        if i['tag'] == tag:
                            bot_response = random.choice(i['responses']) #select a random response
                    
                    await message.channel.send(bot_response.format(message))#sending the message
                
                else:
                    await message.channel.send("I didn't get that. Can you explain or try again.".format(message))

client = MyClient()
client.run('*********************************************') #bot token
