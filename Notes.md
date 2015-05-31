#How I Fight Slavery

By Eric Schles

#About Me

* Developer Evangelist @ Syncano
* Slavery Fighter for the Manhattan District Attorney's Office
* Researcher at NYU

##Introduction and Background

###Problem Definition
Slavery is a serious problem.  It is deeply nuanced and complex. 

####Understanding Scale

* 9,298 unique cases of human trafficking from Human Trafficking hotline (over 5 years)

Source: [Polaris Project 1](http://www.polarisproject.org/human-trafficking/overview/human-trafficking-trends)

####Understanding on a personal level

* [Cracked Article](http://www.cracked.com/personal-experiences-1440-5-things-i-learned-as-sex-slave-in-modern-america.html)
* [Some more examples](http://www.equalitynow.org/survivorstories)

####Slavery Around the World
* [unodc - 2014](http://www.unodc.org/documents/data-and-analysis/glotip/GLOTIP_2014_full_report.pdf)

##Formal Definition - Human Trafficking

Human trafficking := the process by which a person is deprived of rights and forced to work against their will for either very little monetary compensation or no monetary compensation.

###Types of Human Trafficking

* Sex Trafficking (which I will discuss in detail today)
* Labour Trafficking
* Bondage
* Indebted Servant
* Child Sex Trafficking

##Formal Definition - Sex Trafficking

Sex Trafficking := the practice of monetary exploitation from systematic and continual rape of another.  In order to prove sex trafficking has occurred, it must be shown that force, fraud or coercion was used.  

##Building a semi-automated investigative system

* Generating Leads

* Analyzing Leads and Collecting Information for Prosecution

* Making Sense of the Data for a Jury

###Generating Leads

Lead Generation is the process of finding instances of that could be human trafficking.  There are two ways to do this:

* Human Assisted Lead Generation 
* Completely Automated Lead Generation

##Human Assisted

My Human Assisted tool is called [Investagator](https://github.com/EricSchles/investa_gator_v2)

###[Demo Goes Here]

It allows investigators to take existing known trafficking ads and allows them to completely map a network of traffickers on backpage.  The goal of such a mapping is to find all the ads and therefore women associated with a single advertisement, as well as determine all the locations a given trafficker operates.   

If any data is found it is saved to the database for further analysis.

##What data do we pull down:

There are a few attributes we care about:

* Phone numbers
* Polarity of the text
* Subjectivity of the text
* images in the ad
* When it was scraped
* Emails in the ad (ToDo)

##Getting the Phone number

There has been an increasingly escalting obfuscation war between traffickers and law enforcement to hide phone numbers in plain sight.  

I516 ha7ve se7en o7bfu4sca2tio1n l0ike this.

And ob5fu1sixca7tion li7ke th7i7s ThErE as well, fiVe 1 2 4 my measurements are 32 34 32.

And even more weird cases.

So what the hell do you do, to get the phone number - the most important piece of the puzzle:

I've written three functions:

1) turn all the words into numbers.
```
def letter_to_number(self,text):
        text= text.upper()
        text = text.replace("ONE","1")
        text = text.replace("TWO","2")
        text = text.replace("THREE","3")
        text = text.replace("FOUR","4")
        text = text.replace("FIVE","5")
        text = text.replace("SIX","6")
        text = text.replace("SEVEN","7")
        text = text.replace("EIGHT","8")
        text = text.replace("NINE","9")
        text = text.replace("ZERO","0")
        return text
```

2) parse the numbers from the text        
   
```
    def phone_number_parse(self,values):
        phone_numbers = []
        text = self.letter_to_number(values["text_body"])
        phone = []
        counter = 0
        found = False
        possible_numbers = []
        for ind,letter in enumerate(text):
            if letter.isdigit():
                phone.append(letter)
                found = True
            else:
                if found:
                    counter += 1
                if counter > 15 and found:
                    phone = []
                    counter = 0
                    found = False

            if len(phone) == 10 and phone[0] != '1':
                possible_numbers.append(''.join(phone))
                phone = [] #consider handling measurements
            if len(phone) == 11 and phone[0] == '1':
                possible_numbers.append(''.join(phone))
                phone = [] #consider handling measurements
        for number in possible_numbers:
            if self.verify_phone_number(number):
                phone_numbers.append(number)
        return phone_numbers
```
3) most importantly - verify your phone number is correct - thank you twilio!  And thank you Rob Spectre, you wonderful human you.
```
    def verify_phone_number(self,number):
        data = pickle.load(open("twilio.creds","r"))
        r = requests.get("http://lookups.twilio.com/v1/PhoneNumbers/"+number,auth=data)
        if "status_code" in json.loads(r.content).keys():
            return False
        else:
            return True
```

##Running an Investigation

Once we are reasonably convinced we know everything a manual search will yield we can run an investigation.  This allows the investigators time to be spent on other tasks, while new advertisements will be logged - allowing us to completely understand the network of the traffickers.

```

from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob.classifiers import DecisionTreeClassifier as DTC
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

#..snip..

	def doc_comparison(new_document,doc_list):
	    total = 0.0
	    for doc in doc_list:
	        total += consine_similarity(new_document,doc)[1]
	    if total/len(doc_list) > 0.5: #play with this
	        return "trafficking"
	    else:
	        return "not trafficking"
    
	def cosine_similarity(documentA,documentB):
	    docs = [documentA,documentB]
	    tfidf = TfidfVectorizer().fit_transform(docs) 
	    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten() 
	    return cosine_similarities


    def investigate(self):
        
        data = self.scrape(self.base_urls)
        train_crud = CRUD("sqlite:///database.db",Ads,"ads")
        #getting dummy data from http://www.dummytextgenerator.com/#jump
        dummy_crud = CRUD("sqlite:///database.db",TrainData,"training_data")
        train = train_crud.get_all()
        dummy = dummy_crud.get_all()
        t_docs = [elem.text for elem in train_crud.get_all()] #all documents with trafficking
        train = [(elem.text,"trafficking") for elem in train] + [(elem.text,"not trafficking") for elem in dummy]
        cls = []
        #make use of tdf-idf here
        cls.append(NBC(train))
        cls.append(DTC(train))
        for datum in data:
            for cl in cls:
                if cl.classify(datum["text_body"]) == "trafficking":
                    self.save_ads([datum])
            #so I don't have to eye ball things
            if doc_comparison(datum["text_body"],t_docs) == "trafficking":
                self.save_ads([datum])
        time.sleep(700) # wait ~ 12 minutes
        self.investigate() #this is an infinite loop, which I am okay with.
    
```

As you can see, the investigate function will take the ads already classified as trafficking and look for new ads that are similar enough to be considering trafficking as well.  I'm making use of two classification schemes here - Naive Bayesian Classification and Tree Classification.

###Understanding Naive Bayesian Classification
To understand a naive bayesian classifier it's best to understand the steps involved:

1) hand label a set of texts as certain mappings:

```
[ ("Hello there, I'm Eric","greeting"),
  ("Hi there, I'm Jane","greeting"),
  ("Hi, how are you?","greeting"),
  ("Hello","greeting"),
  ("I'm leaving now, Jane","parting"),
  ("Goodbye","parting"),
  ("parting is such sweet sore, but I'm sleepy, so get out.","parting")
]
```
2) Transform each word in the training set so that it can be described as a set of independent proabilities, that map to a given label:

In this case we split each piece of text by mapping the words to numbers: 

```
"Hello there, I'm Eric" -> Freq(Hello) = 1,  Freq(there,) = 1,  Freq(I'm) = 1, Freq(Eric) = 1 ->
prob(Hello) = 1/4, prob(there,) = 1/4, prob(I'm) = 1/4, prob(Eric) = 1/4 
```
We then apply one of a set of functions (typically a [maximium likelihood estimator](http://en.wikipedia.org/wiki/Maximum_likelihood)) to these mathematical quantities and say this maps to the label.  In this case "greeting"

So we can sort of think of this as:

```
transform = f("Hello there, I'm Eric")
g(transform) = "greeting"
```

3) Verification of our model to our liking 

The next step is to provide a verification set that is hand labeled as well.  The model we've constructed from our training set is applied to this verification set and then checked against the expected label.  If the label and the prediction from the model match, we claim the model is correct for this result.

Once we have enough favorable results we are ready to make use of our text classification system for real world problems 

###Making use of Cosine Similarity (to be lazy)

As you can see, we make use of Tf-Idf to and document similarity to assess whether two pieces of text are similar, initially (and throughout).  This avoids the problem of having to hand label results - but is somewhat of a hack for now.  The main reason for doing this is Naive Bayesian Classification isn't useful without enough labeled data, but typically when a trafficking case comes in, we only get 4 or maybe 5 ads.  Of course, the analysts can go back in later when more ads come in, but it's not always easy to find these new ads.  This is because some ads get taken down, either by the owner or by backpage over time.  Therefore, an analyst may miss a new ad pertaining to a specific case, and we cannot afford to let luck dictate evidence collection.  

The larger point here is there needs to be some automatic way of adding in documents that are similar enough.

####Caveat Worth Mentioning

This tool is in alpha and I'm adding new features regularly.  The next step here is to make use of hardcore algorithms.  My guess is I'll settle on SVMs but I haven't been playing around with this long enough to test which algorithm will be most performant.  If you want to understand text classification with serious algorithms I'd highly recommend checking out:

* [This scikit learn example](http://scikit-learn.org/0.11/auto_examples/document_classification_20newsgroups.html)
* [This stack overflow answer](http://stackoverflow.com/questions/19484499/text-mining-with-svm-classifier)

##Analyzing Leads and Collecting Information for Prosecution

Most traffickers also have their own personal websites.  Once we have enough personal information from scraping backpage, we can typically start to find these domains.

##Scraping other websites

Ad sites like backpage all function more or less the same way - they post a ton of ads per day and there are a lot of pages, so you need specialized scrapers.  Fortunately the ads are all very regular, so you only need to write those scrapers once.

However, there are lots of little website that will sometimes be crucial to an investigation.  So I wrote a little general purpose tool that forensicly maps and stores websites.  It will also rescrape them regularly, as content is updated as often as once per day.

[Alert System](https://github.com/EricSchles/alert_system)

###[Demo Goes here] - alert_system (in slaveryStuff)

###Grabbing all the things

At the heart of this tool is a single function - mapper

```
    def mapper(self,url,depth,link_list):
        """Grabs all the links on a given set of pages, does this recursively."""
        if depth <= 0:
            return link_list
        links_on_page = self.link_grab(url)
        tmp = []
        for link in links_on_page:
            if not link in link_list:
                link_list.append(link)
                tmp = self.mapper(link,depth-1,link_list)
                for elem in tmp:
                    if not elem in link_list:
                        link_list.append(elem)
        return link_list
```

The most important piece of this tool is making recursive calls, at each depth call to the mapper.  

`link_grab` grabs all the links from a html page and then heads to the next depth of the page.  Notice that depth is decremented each time the mapper is called.  Typically websites are 3 or 4 deep, meaning we can grab a lot of links, very quickly making use of this method.

###Storing all the things

Once we have everything grabbed I store it by downloading the html and taking the SHA256 hash to ensure that the website is as we downloaded it, for legal proceedings.  

###Sending Alerts

Rather than paying for scriptable email support, a friend of mine found a way to script sending and receiving emails via gmail with python.  Thanks [Bryan Britten](http://www.theranalyst.com/)  

It is HIGHLY recommended that you create a new gmail address if you want to be able to script it, because you need to do the following:

https://www.google.com/settings/security/lesssecureapps

This essentially makes your gmail address acessible by third party applications, which introduces a ton of security issues.  

Now that we can use our gmail to send email, let's look at our Emailer class - this will be used to alert analysts when the website was updated in some critical way. 

```
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

class Emailer:
    
    def __init__(self,addr='',pw="",website=None): #addr is your email address, pw is password
        self.addr = addr
        self.pw = pw
        self.msg = MIMEMultipart('alternative')
        self.receiver = [''] #email to send to
        if website:
            self.msg['Subject'] = "Update to website"+website
        else:
            self.msg['Subject'] = "Update to website"
        self.msg["From"]=self.addr
        self.msg["To"] = ','.join(self.receiver)
        
    def add_message(self,text):
        if type(text) == type(' '):
            tmp = MIMEText(text,'plain')
            self.msg.attach(tmp)
    def send(self):
        s = smtplib.SMTP('smtp.gmail.com', 587) #used because I don't know how to use Outlook SMTP
        s.starttls()
        s.login(self.addr, self.pw)
        s.sendmail(self.addr, self.receiver, self.msg.as_string()) 
        s.close()
    def add_website(self,site):
        self.website = website
        self.msg['Subject'] = "Update to website"+website
```

The important things here are: 

_Login_:
```
s = smtplib.SMTP('smtp.gmail.com', 587) #used because I don't know how to use Outlook SMTP
s.starttls()
s.login(self.addr, self.pw)
s.sendmail(self.addr, self.receiver, self.msg.as_string())
```

And 

_Sending_:

`self.msg.attach(tmp)`

###Making use of Our Library

Now that we can send email programatically, let's make use of it to diff our files:

```
		#any variable with _t is today
        #any variable with _y is yesterday
        for hash_set_t in today_hashes:
            for ind_t,hashing_t in enumerate(hash_set_t):
                name_t = hashing_t.split("000")[1].split(":")[0]
                hash_val_t = hashing_t.split(":")[1]
                for hash_set_y in yesterday_hashes:
                    for ind_y,hashing_y in enumerate(hash_set_y):
                        hash_val_y = hashing_y.split(":")[1]
                        if name_t in hashing_y:
                        	if hash_val_t != hash_val_y:
	                            self.emailer.add_website(site)
	                            self.emailer.add_message("the website was updated")
	                            self.emailer.send()
```

If the files are different we alert the analyst something has changed, which may be of interest.  Due to the nature of the content and the fact that it's stored on google's servers we include as little information as possible in the alert.  

##Complete Automation

The tool compares faces of missing children and prostitutes in ads scrapped from backpage.  Unfortunately I can't show you the full tool 
because it makes use of some non-open data.  

Instead I'll explain the idea here, as well as some pointers to building your own system.  

In addition to scraping backpage (and other websites) for directed searches, I've also written a nice little "API", which allows you to pull down information for further use.  I make use of this data to do comparison against missing children - looking for kids that were co-opted into sex trafficking.

###Doing The Comparison

Fortunately, new missing children are not added nearly as often as backpage ads go up (somewhere on the order of a million ads per day internationally), therefore when a new missing child is added to our dataset, they can be searched against all backpage ads.   

###Building A Search For Images

This explanation is heavily influenced by [this post](http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/)

In general what we are building is known as a CBIR (Content Based Image Retrival) System.  

####How CBIR works

There are 4 steps to defining a CBIR:

* Define your image descriptor - The image descriptor is how you describe images mathematically
* Indexing your dataset - Map all your images to the mathematical transformation
* Define your similarity metric - this can be euclidean distance, cosine distance, or some other distance measure
* Searching - performing the actual search against new images sent to your datastore.

####Defining your Image Descriptor

There are a lot of ways you can define an image descriptor some examples include:

* Color in the Image
* The Shape of an object in the image
* Characterization of the Texture

Of course this is just a few Naive ideas.  There are lots of notions around [facial recognition](http://link.springer.com/chapter/10.1007%2F978-3-319-00969-8_19), [object recognition](http://homes.cs.washington.edu/~shapiro/dagstuhl3.pdf), horse recognition, and [a few others](http://homes.cs.washington.edu/~shapiro/icpr2002.pdf). 

[How to find Horse Recognition](http://grail.cs.washington.edu/theses/YiPhd.pdf)

In order to make use of Color as a Descriptor we'll use the histograms and feature vectors being the Intensity of the picture.  [This post](http://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot) goes over how to do this in python.

####Indexing your dataset

A Naive way to generate your index is to simply write your mapping to a file with key value pairs, where the key is the image name, and the feature vector is the mapping.

####Defining your similarity metric

In the interest of remaining obvious, we'll use the cosine distance between two feature vectors.  

####Searching

In the interest of simplicity we simply use a linear search, iterating through each value in our file and then showing the closest results first (aka having the smallest distance).

##Some Face comparison (mostly for fun)

Despite not being able to show you my tool, I can certainly show you how to do [face comparison](https://github.com/EricSchles/face_compare).  

This repo shows you a whole bunch of utilities that are useful for doing facial recognition.

Here I'll show a first approximation towards image search - doing direct comparison of faces between two images.

###[Demo here] - face compare (in slaveryStuff)
```
import cv2
import cv
import numpy as np
from glob import glob
import os

CONFIDENCE_THRESHOLD = 100.0
ave_confidence = 0
num_recognizers = 3
recog = {}
recog["eigen"] = cv2.createEigenFaceRecognizer()
recog["fisher"] = cv2.createFisherFaceRecognizer()
recog["lbph"] = cv2.createLBPHFaceRecognizer()

#load the data initial file
filename = os.path.abspath("black_widow.jpg")
face = cv.LoadImage(filename, cv2.IMREAD_GRAYSCALE)
face,label = face[:, :], 1

#load comparison face
compare = os.path.abspath("person.jpg")
compare_face = cv.LoadImage(compare, cv2.IMREAD_GRAYSCALE)
compare_face, compare_label = compare_face[:,:], 2

images,labels = [],[]
images.append(np.asarray(face))
images.append(np.asarray(compare_face))
labels.append(label)
labels.append(compare_label)

image_array = np.asarray(images)
label_array = np.asarray(labels)
for recognizer in recog.keys():
    recog[recognizer].train(image_array,label_array)


#generate test data
test_images = glob("testing/*.jpg")
test_images = [(np.asarray(cv.LoadImage(img,cv2.IMREAD_GRAYSCALE)[:,:]),img) for img in test_images]
for t_face,name in test_images:
    t_labels = []
    for recognizer in recog.keys():
        [label, confidence] = recog[recognizer].predict(t_face)
        print "match found",name, confidence, recognizer
```

The magic more or less happens here: 

```
for recognizer in recog.keys():
        [label, confidence] = recog[recognizer].predict(t_face)
        print "match found",name, confidence, recognizer
```

As the classifiers show, the predict method returns a simple distance metric where 0.0 means the faces are the same, and anything larger implies difference.  Typically within 50 is a pretty good match.  Of course, the different recognizers will yield different ranges of accuracy and correctness.  Please do tune accordingly.

##Making Sense of the Data for a Jury

There is a great challenge in explaining all of this information to non-technical folks.  Often data visualization, high level statistics like:

From 2011-2014 Person-X posted 211 ads and beat girls in six of these ads, as evidenced by these police reports.

When going to trial, lawyers will often question the authenticity of our data, which is why all websites are hashed.  Another important tool is being able to analyze pictures forensically, for exif data and stegnographic information.

Since this is a data science conference I'll simply [link to the github](https://github.com/EricSchles/picture_forensics), which you can check out if you are interested.

##Questions?

Thanks!

Contact info:

eric.schles@syncano.com
