from nltk.corpus import wordnet as wn



def get_senses(word):
    """
    Extracts all gloss information from WordNet
    :param word: the word in the input sentence
    :return: all WN synsets of the word, their definitions, examples as well as hypernym and hyponym relations
    """
    synsets = wn.synsets(word)
    gloss_synsets = []
    for i in range(len(synsets)):
        syn = {
                "synset": synsets[i],
                "name": synsets[i].name(),
                "offset": synsets[i].offset(),
                "definition": synsets[i].definition(),
                "examples": synsets[i].examples(),
                "hypernyms": synsets[i].hypernyms(),
                "hyponyms": synsets[i].hyponyms()
              }
        gloss_synsets.append(syn)
        # I changed "name" to "stem_word"
    word_synsets = {"word": {"stem_word": word}, "synsets": gloss_synsets}
    return word_synsets


bass = get_senses("bass")
fish = get_senses("fish")
saltwater_fish = get_senses("saltwater_fish")
freshwater_fish = get_senses("freshwater_fish")

print(bass)
print(fish)
print(saltwater_fish)
print(freshwater_fish)