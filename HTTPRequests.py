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

    def onInput_onStart(self):
        audio = ALProxy("ALAudioDevice")
        tts = ALProxy("ALTextToSpeech")
        record = ALProxy("ALAudioRecorder")
        aup = ALProxy("ALAudioPlayer")
        tts.say("What would you like to search?")
        record_path = 'record.wav'
        record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
        time.sleep(3)
        record.stopMicrophonesRecording()

        r = sr.Recognizer()
        with sr.AudioFile("record.wav") as source:
            audio_data = r.record(source)
            output = r.recognize_google(audio_data)
        tts.say(str(output))
        if(str(output) == "movie"):
            tts.say("What movie would you like to hear about?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
            imdblink = self.IMDBSearch(str(output))
            result = self.getDescription(imdblink)

        elif(str(output) == "word"):
            tts.say("What word should I look up for you?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
            result = self.searchDictionary(str(output))

        elif(str(output) == "recipe"): #recipe
            tts.say("What dish would you like to hear about?")
            record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
            time.sleep(2)
            record.stopMicrophonesRecording()
            with sr.AudioFile("record.wav") as source:
                audio_data = r.record(source)
                output = r.recognize_google(audio_data)
            search = str(output)
            split = search.split(" ")
            searchTerm = ""
            for i in range(0, len(split)):
                if (i == len(split) - 1):
                    searchTerm += split[i]
                    break
                searchTerm += split[i]
                searchTerm += "%20"
            recipeURL = self.getRecipeURL(searchTerm)
            result = self.getRecipe(recipeURL)
            result = result + self.getDirections(recipeURL)
        else:
            tts.say("Sorry, I didn't quite understand that.")


        if(len(result) > 0):
            tts.say(result)
        pass
    def cleanString(self, input):

        output = input.encode('ascii', 'ignore')
        return output;
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

    def searchDictionary(self, word):

        page = requests.get("https://www.merriam-webster.com/dictionary/" + word)
        soup = BeautifulSoup(page.content, "html.parser")
        myDiv = soup.find("span", class_="dtText")
        try:
            for child in myDiv.find_all("span"):
                child.decompose()
        except:
            print("No definition found")
            return
        output = str(myDiv.text)
        output = output.replace(':', '')
        return str.lstrip(output)

    def getRecipeURL(self, foodItem):

        page = requests.get("https://www.allrecipes.com/search/results/?wt=" + foodItem)
        soup = BeautifulSoup(page.content, "html.parser")
        recipeCard = soup.find("div", class_="grid-card-image-container")
        links = recipeCard.find_all('a', href=True)
        try :
            #print(links[0].get('href'))
            return links[0].get('href')
        except:
            print("No recipe found with that name")
            return ""

    def getRecipe(self, url):

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        mySection = soup.find("div", class_="recipe-shopper-wrapper")
        ingredientList = ""
        try:
            ingredients = mySection.find_all("span", class_="ingredients-item-name")
            for ingredient in ingredients:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "
        except:
            mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-1")
            ingredients = mySection.find_all("span", class_="recipe-ingred_txt added")
            mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-2")
            ingredients2 = mySection.find_all("span", class_="recipe-ingred_txt added")
            for ingredient in ingredients:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "
            for ingredient in ingredients2:
                nextLine = str.lstrip(self.cleanString(ingredient.text))
                ingredientList = ingredientList + nextLine + ".\n "

        return ingredientList



    def getDirections(self, url):

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        mySection = soup.find("section", class_="recipe-instructions recipe-instructions-new component container")
        directionList = ""
        try:
            directions = mySection.find_all("div", class_="section-body")
            #print("----------Directions------------")
            for i in range(0, len(directions)):
                step = i + 1
                nextLine = str.lstrip(self.cleanString(directions[i].text))
                directionList = directionList + "Step " + str(step) + ": " + nextLine + ".\n "
        except:
            mySection = soup.find("div", class_="directions--section")
            directions = mySection.find_all("span", class_="recipe-directions__list--item")
            #print("----------Directions------------")
            for i in range(0, len(directions)):
                step = i + 1
                nextLine = str.lstrip(self.cleanString(directions[i].text))
                if(len(nextLine) == 0):
                   break
                else:
                   directionList = directionList + "Step " + str(step) + ": " + nextLine + ".\n "
        return directionList

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
