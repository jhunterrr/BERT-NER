from bert import Ner
import sys

# model = Ner(str(sys.argv))
model = Ner("./content/out_base_simp_shuffled/")
path = "BERT-NER/test.py/data/valid.txt"

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
model.evaluate_zero_shot(path, ["person", "location", "organisation"])

