import speech_recognition as sr
from naoqi import ALProxy
import time
import requests
from bs4 import BeautifulSoup
class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)

    def onLoad(self):
        #put initialization code here
        pass

    def onUnload(self):
        #put clean-up code here
        pass
    
    #Record the audio for the question and direct the question to the appropriate method
    def onInput_onStart(self):
    
        #initializing variables for input and output
        audio = ALProxy("ALAudioDevice")
        tts = ALProxy("ALTextToSpeech")
        record = ALProxy("ALAudioRecorder")
        aup = ALProxy("ALAudioPlayer")
        
        #Ask the user what they would like to search and record their response
        tts.say("What would you like to search?")
        record_path = 'record.wav'
        record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
        time.sleep(3)
        record.stopMicrophonesRecording()

        #Definition for speech recognition
        r = sr.Recognizer()
        
        #Process the audio file via google's speech recognition library using our recorded wav file as the source
        with sr.AudioFile("record.wav") as source:
            audio_data = r.record(source)
            output = r.recognize_google(audio_data)
        
        
        #If the audio output was movie, ask the user what movie they'd like to hear about, record the response and direct the output to the appropriate method
        if(str(output) == "movie"):
            tts.say("What movie would you like to hear about?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
                
            #Get the IMDB link for the movie
            imdblink = self.IMDBSearch(str(output))
            
            #Scrape the description of the movie
            result = self.getDescription(imdblink)

        #If the output is a word, ask the user for the word they'd like to search, store and process the audio, and send the output to the appropriate method
        elif(str(output) == "word"):
            tts.say("What word should I look up for you?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
                
            #Scrape the definition of the word from merriam webster and store in result
            result = self.searchDictionary(str(output))

        #If the output is recipe, ask the user for a dish, store and process the audio file and send the output to the appropriate method
        elif(str(output) == "recipe"): #recipe
            tts.say("What dish would you like to hear about?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
            search = str(output)
            
            #Process the output into a URL as formatted by AllRecipe's site
            split = search.split(" ")
            searchTerm = ""
            for i in range(0, len(split)):
                if (i == len(split) - 1):
                    searchTerm += split[i]
                    break
                searchTerm += split[i]
                searchTerm += "%20"
            
            #Get the Recipe URL
            recipeURL = self.getRecipeURL(searchTerm)
            
            #Scrape the recipe and store in result
            result = self.getRecipe(recipeURL)
            
            #Append the instructions to the end of the ingredients
            result = result + self.getDirections(recipeURL)
            
        #If the input wasn't recognized, let the user know 
        else:
            tts.say("Sorry, I didn't quite understand that.")

        #If result isn't empty, say the result
        if(len(result) > 0):
            tts.say(result)
        pass
        
    #Method that removes ascii characters from output to make it more readable for tts.say
    def cleanString(self, input):

        output = input.encode('ascii', 'ignore')
        return output;
        
    #Method that grabs HTML from the requested website and confirms the requested movie link exists
    def IMDBSearch(self, term):

        #requests.get loads the html into the variable "page"
        page = requests.get("https://www.imdb.com/find?ref_=nv_sr_fn&q=" + term + "&s=all")
        #soup parses the html and makes it searchable
        soup = BeautifulSoup(page.content, "html.parser")
        #myDivs stores the information from the class "findSection" in the html
        myDivs = soup.find("div", class_="findSection")
        #links further refines this search and takes only the href starting with "title" and stores it in links
        try :
            links = myDivs.select_one("a[href*=title]")
        except:
            print("No movie found with that title")
            return ""
        #this will return the first of the links stored in links
        return("imdb.com" + links.get('href'))

        
    #Method that uses the link as given above to scrape the HTML for the requested information
    def getDescription(self, link):

        #store the page of the given link. The above method returns everything after "https://www.", so we must precede with that text
        page = requests.get("https://www." + link)
        #again, store the html in soup
        soup = BeautifulSoup(page.content, "html.parser")
        #find the summary
        myDiv = soup.find("div", class_="summary_text")
        myTitle = soup.find("div", class_="title_wrapper")
        for child in myTitle.find_all("div"):
            child.decompose()
        summary = ""
        summary = summary + str.lstrip(self.cleanString(myTitle.text))
        #strip the preceding spaces
        summary = summary + str.lstrip(str(myDiv.text))
        #print the summary
        return summary

    #Method that stores the HTML of the requested word on meriam-webster.com and scrapes the html for the definition of the word
    def searchDictionary(self, word):

        #Grab the HTML for the word
        page = requests.get("https://www.merriam-webster.com/dictionary/" + word)
        
        #Define the HTML scraper
        soup = BeautifulSoup(page.content, "html.parser")
        
        #Scrape the html for the specific class 'dtText' which contains our definition
        myDiv = soup.find("span", class_="dtText")
        try:
            for child in myDiv.find_all("span"):
                child.decompose()
        except:
            print("No definition found")
            return
        #Convert the output of the above scrape to a string
        output = str(myDiv.text)
        
        #Get rid of the ':' in the definition
        output = output.replace(':', '')
        return str.lstrip(output)

    #Method that uses the requested dish to gather the HTML from allrecipes.com
    def getRecipeURL(self, foodItem):

        #Store the html from the given item on AllRecipes.com in page
        page = requests.get("https://www.allrecipes.com/search/results/?wt=" + foodItem)
        
        #Initialize the scraper
        soup = BeautifulSoup(page.content, "html.parser")
        
        #Find the specific class 'grid-card-image-container' which holds all of the results on the first page for the search
        recipeCard = soup.find("div", class_="grid-card-image-container")
        links = recipeCard.find_all('a', href=True)
        
        #Return the link for the very first recipe in our search
        try :
            #print(links[0].get('href'))
            return links[0].get('href')
        except:
            print("No recipe found with that name")
            return ""

    #Using the above link, this method gathers the HTML from the dish's page, stores and processes the HTML, then returns the ingredient list for the dish
    def getRecipe(self, url):

        #Grab and store the HTML in page
        page = requests.get(url)
    
        #Initialize the scraper
        soup = BeautifulSoup(page.content, "html.parser")
        
        #Find the specific class 'recipe-shopper-wrapper' which contains our ingredient list
        mySection = soup.find("div", class_="recipe-shopper-wrapper")
        
        #Initialize our ingredient list
        ingredientList = ""
        
        #Try catch for differentiating between desserts and main dishes. Desserts are stored differently and will run into an error when trying to process
        #in the same way as main dishes
        
        try:
            #Find the class 'ingredients-item-name' which contains our ingredients
            ingredients = mySection.find_all("span", class_="ingredients-item-name")
            
            #For each ingredient, strip the line of preceding spaces and store the ingredient in ingredientList
            for ingredient in ingredients:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "
                
        except:
            #Find the split section of ingredients found in 'checlist dropdownwrapper list-ingredients-1(or 2)'
            
            mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-1")
            ingredients = mySection.find_all("span", class_="recipe-ingred_txt added")
            mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-2")
            ingredients2 = mySection.find_all("span", class_="recipe-ingred_txt added")
            #for each ingredient in both lists, strip the preceeding spaces and store in ingredientList
            for ingredient in ingredients:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "
            for ingredient in ingredients2:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "

        return ingredientList


    #Using the same link as above, gather and store the instruction list from the HTML for the given dish
    def getDirections(self, url):

        #Store the HTML of the page in 'page'
        page = requests.get(url)
        
        #Initialize the scraper
        soup = BeautifulSoup(page.content, "html.parser")
        
        #Find the class 'recipe-instructions recipe-instructions-new component container' which contains our instruction list
        mySection = soup.find("section", class_="recipe-instructions recipe-instructions-new component container")
        directionList = ""
        
        #Once again, instrcutions are stored differently with different kinds of dishes so try the first one and if you run into an error, try the second format 
        #(All dishes are stored in one of these two ways)
        try:
            #Find the class 'section-body' which will contain our instruction list or throw an error and move us into the except
            directions = mySection.find_all("div", class_="section-body")
            #print("----------Directions------------")
            #For each direction, strip preceeding spaces and store in directionList
            for i in range(0, len(directions)):
                step = i + 1
                nextLine = str.lstrip(self.cleanString(directions[i].text))
                directionList = directionList + "Step " + str(step) + ": " + nextLine + ".\n "
        except:
            #Look for the specific class 'directions--section'
            mySection = soup.find("div", class_="directions--section")
            directions = mySection.find_all("span", class_="recipe-directions__list--item")
            #print("----------Directions------------")
            #For each direction, strip preceeding spaces and store in directionList
            for i in range(0, len(directions)):
                step = i + 1
                nextLine = str.lstrip(self.cleanString(directions[i].text))
                if(len(nextLine) == 0):
                   break
                #add 'step i' to each step to make reading the directions more clear
                else:
                   directionList = directionList + "Step " + str(step) + ": " + nextLine + ".\n "
        return directionList

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
