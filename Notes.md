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

##Why all this data

Once the data is collected it is compared against internal data sets and other records which are used to map the life of the trafficker - looking for anything to convict them of a crime.  Proving trafficking is extremely hard - this is because often times the victims will recant their testimoney, several times.  This is sometimes due to fear of further abuse, or even death.

##Complete Automation

The tool compares faces of missing children and prostitutes in ads scrapped from backpage.  Unfortunately I can't show you the full tool because