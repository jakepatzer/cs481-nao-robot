import requests
from bs4 import BeautifulSoup
import re


#param "term" is the formatted search criteria for IMDB

def IMDBSearch(term):

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

def getDescription(link):

    #store the page of the given link. The above method returns everything after "https://www.", so we must precede with that text
    page = requests.get("https://www." + link)
    #again, store the html in soup
    soup = BeautifulSoup(page.content, "html.parser")
    #find the summary
    myDiv = soup.find("div", class_="summary_text")
    myTitle = soup.find("div", class_="title_wrapper")
    for child in myTitle.find_all("div"):
        child.decompose()
    print(myTitle.text)
    #strip the preceding spaces
    summary = str.lstrip(myDiv.text)
    #print the summary
    print(summary)

def searchDictionary(word):

    page = requests.get("https://www.merriam-webster.com/dictionary/" + word)
    soup = BeautifulSoup(page.content, "html.parser")
    myDiv = soup.find("span", class_="dtText")
    try:
        for child in myDiv.find_all("span"):
            child.decompose()
    except:
        print("No definition found")
        return
    print(myDiv.text)

def getRecipeURL(foodItem):

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

def getRecipe(url):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    mySection = soup.find("div", class_="recipe-shopper-wrapper")
    try:
        ingredients = mySection.find_all("span", class_="ingredients-item-name")
        print("----------Ingredients------------")
        for ingredient in ingredients:
            nextLine = str.lstrip(ingredient.text)
            print(nextLine)
    except:
        mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-1")
        ingredients = mySection.find_all("span", class_="recipe-ingred_txt added")
        mySection = soup.find("ul", class_="checklist dropdownwrapper list-ingredients-2")
        ingredients2 = mySection.find_all("span", class_="recipe-ingred_txt added")
        print("----------Ingredients------------")
        for ingredient in ingredients:
            nextLine = str.lstrip(ingredient.text)
            print(nextLine)
        for ingredient in ingredients2:
            nextLine = str.lstrip(ingredient.text)
            print(nextLine)



def getDirections(url):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    mySection = soup.find("section", class_="recipe-instructions recipe-instructions-new component container")
    try:
        directions = mySection.find_all("div", class_="section-body")
        print("----------Directions------------")
        for i in range(0, len(directions)):
            step = i + 1
            nextLine = str.lstrip(directions[i].text)
            print("Step " + str(step) + ": " + nextLine)
    except:
        mySection = soup.find("div", class_="directions--section")
        directions = mySection.find_all("span", class_="recipe-directions__list--item")
        print("----------Directions------------")
        for i in range(0, len(directions)):
            step = i + 1
            nextLine = str.lstrip(directions[i].text)
            if(len(nextLine) == 0):
               break
            else:
               print("Step " + str(step) + ": " + nextLine)

def main():

    #Take in input from the user and format it as required for IMDB
    #Example: it chapter 2 -> it+chapter+2
    searchType = input("What would you like to search?: Movie, Word or Recipe ")
    str.capitalize(searchType)
    if (searchType.lower() == 'movie'):
        search = input("Search something: ")
        split = search.split(" ")
        searchTerm = ""
        for i in range(0, len(split)):
            if (i == len(split) - 1):
                searchTerm += split[i]
                break
            searchTerm += split[i]
            searchTerm += "+"
        #gets the URL for the movie we're searching for
        movieLink = IMDBSearch(searchTerm)
        #prints the description of the movie we searched for
        if (len(movieLink) > 0):
            getDescription(movieLink)
    elif (searchType.lower() == 'word'):

        search = input("Enter a word: ")
        searchDictionary(search)

    else:

        search = input("Enter a dish you'd like to hear about: ")
        split = search.split(" ")
        searchTerm = ""
        for i in range(0, len(split)):
            if  (i == len(split) - 1):
                searchTerm += split[i]
                break
            searchTerm += split[i]
            searchTerm += "%20"

        recipeURL = getRecipeURL(searchTerm)
        getRecipe(recipeURL)
        getDirections(recipeURL)

if __name__ == "__main__":
    main()
