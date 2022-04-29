class Loc:

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __eq__(self, other):
        return self.latitude == other.latitude and self.longitude == other.longitude

    def __hash__(self):
        return hash((self.latitude, self.latitude))  # str(self.latitude)+str(self.latitude)


item = Loc(1, 2)
myDict = {item: 10}
item1 = Loc(1, 2)
print(myDict[item1])
