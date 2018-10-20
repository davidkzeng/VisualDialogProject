import json
from sklearn.feature_extraction.text import CountVectorizer

train_file = '../dialogs/visdial_1.0_train.json'
train_file_json = None

with open(train_file) as json_data:
    train_file_json = json.load(json_data)
    json_data.close()

train_file_json = train_file_json['data']
dialogs = train_file_json['dialogs']
captions = [dialog['caption'] for dialog in dialogs]

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english', max_df=0.1).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# common_words = get_top_n_words(captions, 300)
# for word, freq in common_words:
#     print(word, freq)

# extract_words = ['tennis', 'plate']
# output_file = '../dialogs/random_images.json'
extract_words = ['dog', 'cat', 'bear', 'horse', 'sheep', 'giraffe']
output_file = '../dialogs/more_animals.json'
# extract_words = ['snow', 'beach', 'city', 'sky', 'trees']
# output_file = '../dialogs/nature.json'

labeled_images = {}

for dialog in dialogs:
    caption = dialog['caption']
    image_num = dialog['image_id']
    if image_num == 25:
        print(caption)
    lowercase = caption.lower()
    num_appear = 0
    for i, extract_word in enumerate(extract_words):
        if extract_word in caption:
            if image_num == 46223:
                print(caption, extract_word)
            labeled_images[image_num] = i
            num_appear += 1
    if num_appear > 1:
        del labeled_images[image_num]        

data = {}
data['classes'] = extract_words
data['labeled_images'] = labeled_images
with open(output_file, 'w+') as outfile:
    json.dump(data, outfile)
