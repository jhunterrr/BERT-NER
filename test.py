from bert import Ner
import sys

def evaluate_zero_shot(filename: str, label_list=None: list):
    simplified_labels = { "O": "O", "B-MISC": "miscellaneous", "I-MISC": "miscellaneous", "B-PER": "person", "I-PER": "person", 
                         "B-ORG": "organisation", "I-ORG": "organisation", "B-LOC": "location", "I-LOC": "location", "[CLS]": "[CLS]", "[SEP]": "[SEP]" } 
    #initialise text and value for retrieving label
    text = []
    ground_truth = []
    entities_selected = 0
    entities_relevant = 0
    true_positives = 0
    total_entities = 0
    label_loc = 4
    #read file
    with open(filename) as file:
        next(file)
        for line in file:
                #if blank line, process method and reset sentence
                if line.isspace() and text != []:
                    for i, old_lab in enumerate(ground_truth):
                        #print(simplified_labels[old_lab.strip()])
                        ground_truth[i] = simplified_labels[old_lab.strip()]
                    #print(ground_truth)
                    if label_list != None:
                        tp, es, er = model.predict_zero_shot(text, label_list, ground_truth)
                    else:
                        tp, es, er = model.predict_original(text, ground_truth)
                    true_positives += tp
                    entities_selected += es
                    entities_relevant += er
                    text = []
                    ground_truth = []
                #if not blank line, add to line, find label and assign it
                if not line.isspace():
                    word = line.split(' ')[0]
                    label = line.split(' ')[label_loc-1]
                    if label.strip() != 'O':
                        total_entities += 1
                    text.append(word)
                    ground_truth.append(label)
          accuracy = true_positives / total_entities
        recall = true_positives / entities_relevant
        precision = true_positives / entities_selected
        f1 = 2 * (precision * recall) / (precision + recall)
        
        #metric evaluation
        print("Accuracy: " + str(accuracy))
        print("Recall: " + str(recall))
        print("Precision: " + str(precision))
        print("F1 score: " + str(f1))

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
expected = model.predict("location")
output = model.predict("Berlin New York Munich Belfast Dublin England America")
print(expected)
print("Actual Results:")
print(output)

#New method
print("Group 4: ZER Test")
#print("EVALUATION: OUR BERT METHOD")
#evaluate_zero_shot(path, ["person", "location", "organisation"])
        
print("EVALUATION: ORIGINAL BERT METHOD")
#model = Ner("/content/drive/MyDrive/Colab Notebooks/BERT-NER/out_base")
#evaluate_zero_shot(path)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_dir_orig",
                        default=None,
                        type=str,
                        required=True,
                        help="The original BERT model to be used in evaluation")
    parser.add_argument("--model_dir_new", default=None, type=str, required=True,
                        help="The new BERT model to be used in evaluation ")
    
    
    print("EVALUATION: OUR BERT METHOD")
    model = Ner(args.model_dir_new, ["person", "location", "organisation", "miscellaneous"])
    evaluate_zero_shot(path)
    
    print("EVALUATION: ORIGINAL BERT METHOD")
    model = Ner(args.model_dir_orig)
    evaluate_zero_shot(path)




