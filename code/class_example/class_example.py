# class for a dog object
class Dog:
    
    # initialization method with internal data
    def __init__(self, petname, tempr):
        self.name = petname
        self.temperature = tempr
        
    # get status
    def status(self):
        print("The dog's name is", self.name)
        print("The dog's temperature is", self.temperature, "°C")
    
    # set temperature
    def setTemperature(self, tempr):
        self.temperature = tempr

    # let the dog bark
    def bark(self):
        print("woof! woof!")
    

# Create a new dog object from the Dog class
snowie = Dog("Snowie", 38)
snowie.status()

snowie.setTemperature(40)
snowie.status()
