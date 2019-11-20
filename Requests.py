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

def main():

    #Take in input from the user and format it as required for IMDB
    #Example: it chapter 2 -> it+chapter+2
    searchType = input("What would you like to search?: Movie or Word ")
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
    else:

        search = input("Enter a word: ")
        searchDictionary(search)

if __name__ == "__main__":
    main()
