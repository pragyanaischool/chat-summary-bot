import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import traceback
import sys
import random
from Centroid import Centroid
import math
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmeans import kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from rake_nltk import Metric, Rake
import re
from pytz import timezone

# How many key phrases/topics are gathered per conversation 
NUM_PHRASES_PER_CONV = 4
# The channel we want our bot to send messages in
CHANNEL_ID = '404038433250607104'

# How the bot is summoned
Client = discord.Client();
client = commands.Bot(command_prefix='bopas ')

# Bootup confirmation
@client.event
async def on_ready():
    print("BOT READY")

# Returns key phrases from the chat log based on the amount of hours, conversation and phrases specified
@client.command(pass_context=True)
async def getSummary(ctx):
    # Ignores messages that bots send to avoid infinite loops
    if ctx.message.author == client.user: return
    if ctx.message.author.bot: return
    try:
        # Parameter one in command is how many hours to summarize
        timeToCheck = int(ctx.message.content.split(" ")[2])
        # Parameter two is asking how many conversations the user wants the history split into
        numberOfConversations = int(ctx.message.content.split(" ")[3])
        if timeToCheck <= 18: # Limits how far back the user can summarize
            data = []         # Will hold the time each message was sent - used for clumping
            messagesBin = []  # Will hold each message sent - used to gather messages after being clumped
            currentTime = datetime.utcnow()
            earliestTime = currentTime - timedelta(hours = timeToCheck) # Time when the last message we will look at was sent
            async for singleMessage in client.logs_from(ctx.message.channel, limit = 10000000):
                if singleMessage.timestamp > earliestTime: # If message is outside of our time frame, stop looking through the log
                    messagesBin.append(singleMessage.content) # Add data to our arrays
                    data.append([int((singleMessage.timestamp - earliestTime).seconds), 1]) # timestamp data for clumping
                else:
                    print("Reached end of alloted time period")
                    break
            await client.send_message(client.get_channel(CHANNEL_ID), 
            # Send the key phrases of each conversation
            embed=displayData(analyzeConversations(clusterData(data, numberOfConversations, True), 
            messagesBin), timeToCheck, clusterData(data, numberOfConversations, False), datetime.now(timezone('US/Eastern'))))
    except Exception: 
        print(traceback.format_exc())

# Takes our message timestamp data and clumps the messages into conversations based on
# number of conversations specified by the user. Uses K-Means clumping algorithim. 
# https://www.slideshare.net/AndreiNovikov1/pyclustering-tutorial-kmeans
def clusterData(data, numberOfConversations, getData):
    initial_centers = kmeans_plusplus_initializer(data, numberOfConversations).initialize()
    instance = kmeans(data, initial_centers)
    instance.process()
    #kmeans_visualizer.show_clusters(data, instance.get_clusters(), instance.get_centers(), initial_centers)
    if getData:  # returns list that specifies which messages belong to which conversation
        return instance.get_clusters()
    else:        # returns time list of when each conversation occured
        return instance.get_centers()

# Uses RAKE natural language processing to detect important phrases in a conversation 
# https://pypi.org/project/rake-nltk/
def analyzeConversations(data, messages): 
    ans = []
    for i in data:  # Loops through conversations aka message clusters
        r = Rake()
        conversationString = "" # Will hold every message in a conversation clump
        for j in i: # Loops through messages in conversation (indexes in messages array)
            conversationString += parseString(messages[j]) + " "
        r.extract_keywords_from_text(conversationString) # Gets key phrases
        phrases = r.get_ranked_phrases()
        convoPhrases = []
        for z in range(NUM_PHRASES_PER_CONV): # Get the amount of phrases the user specifies
            if z >= len(phrases):
                break
            convoPhrases.append(phrases[z])
        ans.append(convoPhrases)
    return ans

# Formats the message that displays the data 
def displayData(data, numHrs, centers, time):
    embed = discord.Embed(
        description="Conversation Topics from the last " + str(numHrs) + " hours:",
        color=0x00ff00
    )
    count = 0
    for i in data: # data simply holds the phrases we are outputting from each conversation
        phrases = "" 
        for j in range (NUM_PHRASES_PER_CONV):
            if j >= len(i): 
                break
            phrases += i[j] + "\n"

        timeOfMessage = time - timedelta(hours = int(numHrs - centers[count][0] / 3600), minutes = int(60 - centers[count][0] / 60)) 
        d = timeOfMessage.strftime("%Y-%m-%d %H:%M:%S") # 24 hour time to 12 hour time
        d = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %I:%M:%S %p")
        d = d.split(" ")
        embed.add_field(name ="Conversation " + str(count+1) + ". At " + d[1] + d[2], value = phrases)
        count += 1
    embed.set_thumbnail(url="https://cdn.discordapp.com/attachments/140564130489696256/478291283190743052/unknown.png")
    return embed

# Sees if a string in a message is valid
# We don't want to get rid of some words like 'it' or 'the' as it used in the NLP extraction.
# We do filter based on some common words that I see reoccuring that aren't insightful
# Removes words with any non-ascii characters
def parseString(stringIn):
    pattern = re.compile("[A-z0-9.,!]")
    ans = ""
    for s in stringIn.split(" "):
        if s == "":
            break
        elif not pattern.match(s):
            break
        elif ":" in s:
            break
        elif "@" in s:
            break
        if s in open('wordList.txt').read():
            break
        else:
            ans += s + " "
    return ans
    
client.run("TOKEN");  
