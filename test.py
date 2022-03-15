from bert import Ner
import sys

model = Ner(sys.argv)

#Persons
print("Group 1: Persons")
print("Expected: Final classification for person label below")
expected = model.predict("person")
output = model.predict("Steve John James Daniel Zendaya Robert Pattinson")
print(expected)
print("Actual Results:")
print(output)
  
#Organisations
print("Group 2: Organisations")
print("Expected: Final classification for person label below")
expected = model.predict("organisation")
output = model.predict("Starbucks Microsoft AMD Intel Target Kellogs Walmart Disney")
print(expected)
print("Actual Results:")
print(output)

#Locations
print("Group 3: Locations")
print("Expected: Final classification for person label below")
expected = model.predict("Locations")
output = model.predict("Berlin New York Munich Belfast Dublin England America")
print(expected)
print("Actual Results:")
print(output)

#New method
print("Group 4: ZER Test")
model.predict_zero_shot("Steve goes to the shop in Belfast, Apple", ["person", "location", "organisation"])
output = model.predict("Steve goes to the shop in Belfast, Apple, [SEP] person location organisation")
print(output)
