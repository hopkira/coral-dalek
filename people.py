import pickle

class Person:
    def __init__(self, name):
        self.name = name

f = open("labels.pickle", 'rb')
labels = pickle.load(f)
labels_set = set(labels)
labels = list(labels_set)
known_people = []
for label in labels:
    known_people.append(Person(label))

faces = ["richard", "trevor", "unknown", "kirsty"]

for face in faces:
    for search_face in known_people : 
        if search_face.name == face :
            print("Just seen", search_face.name)