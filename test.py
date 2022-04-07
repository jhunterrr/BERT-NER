from bert import Ner
import sys

def evaluate_zero_shot(filename: str, label_list: list):
    simplified_labels = { "O": "O", "B-MISC": "miscellaneous", "I-MISC": "miscellaneous", "B-PER": "person", "I-PER": "person", 
                         "B-ORG": "organisation", "I-ORG": "organisation", "B-LOC": "location", "I-LOC": "location", "[CLS]": "[CLS]", "[SEP]": "[SEP]" } 
    #initialise text and value for retrieving label
    text = []
    ground_truth = []
    entities_selected = 0
    entities_relevant = 0
    true_positives = 0
    label_loc = 4
    #read file
    with open(filename) as file:
        next(file)
        for line in file:
                #if blank line, process method and reset sentence
                if line.isspace() and text != []:
                    for i, old_lab in enumerate(ground_truth):
                        print(simplified_labels[old_lab.strip()])
                        ground_truth[i] = simplified_labels[old_lab.strip()]
                    print(ground_truth)
                    true_positives, entities_selected, entities_relevant += model.predict_zero_shot(text, label_list, ground_truth)
                    text = []
                    ground_truth = []
                #if not blank line, add to line, find label and assign it
                if not line.isspace():
                    word = line.split(' ')[0]
                    label = line.split(' ')[label_loc-1]
                    text.append(word)
                    ground_truth.append(label)
         print("Recall: " + str(true_positives / entities_relevant))
         print("Precision: " + str(true_positives / entities_selected))

# model = Ner(str(sys.argv))
model = Ner("/content/drive/MyDrive/Colab Notebooks/BERT-NER/out_base_simp_shuffled/content/out_base_simp_shuffled")
path = "/content/drive/MyDrive/Colab Notebooks/BERT-NER/valid.txt"

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
evaluate_zero_shot(path, ["person", "location", "organisation"])





