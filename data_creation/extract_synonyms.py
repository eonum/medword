"""
This file extracts synonym-sets from the ETD.TXT file.
- A synonym-set is a bunch of phrases (can be more than one word per phrase)
  from which any two are synonyms.
- Each set is printed onto a single line, synonym-phrases separated by ';'
"""

import csv
def main():
    input_file_name = "ETD.TXT"
    output_file_name = "mesh_synonyms3.csv"
    synonyms = {}
    used_words = {}
    with open(input_file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[1] not in synonyms:
                synonyms[row[1]] = []
            word = row[0].lower().replace(',', '')

            if 'ß' in word:
                word = word.replace('ß', 'ss')

            if word not in used_words:
                synonyms[row[1]].append(word)
                used_words[word] = 1

    for key in synonyms:
        for word in synonyms[key]:
            if ('ae' in word) or ('ue' in word) or ('oe' in word):
                w2 = word.replace('ae', 'ä')
                w2 = w2.replace('oe', 'ö')
                w2 = w2.replace('ue', 'ü')

                if w2 in synonyms[key]:
                    while word in synonyms[key]: synonyms[key].remove(word)


    output = [[s for s in synonyms[k]] for k in synonyms if len(synonyms[k]) > 1] 
 
    with open(output_file_name, "w") as f:
        writer = csv.writer(f, delimiter=';', lineterminator="\n")
        writer.writerows(output)

if __name__ == "__main__":
    main()
