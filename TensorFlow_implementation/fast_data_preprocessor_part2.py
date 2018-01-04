'''
    This script picks up from where we left in the first part.
'''

from __future__ import print_function
from Summary_Generator.Text_Preprocessing_Helpers.pickling_tools import *
from Summary_Generator.Text_Preprocessing_Helpers.utils import *


# obtain the data from the pickle file generated as an entailment of the preprocessing part 1.
temp_pickle_file_path = "temp.pickle"

# set the limit on the samples to be trained on: 
limit = 600000 # no limit for now

# unpickle the object from this file
print("unpickling the data ...")
temp_obj = unPickleIt(temp_pickle_file_path)

# extract the three lists from this temp_obj
field_names = temp_obj['fields'][:limit]
content_words = temp_obj['content'][:limit]
label_words = temp_obj['label'][:limit]

# print first three elements from this list to verify the sanity:
print("\nField_names:", field_names[: 3]); print("\nContent_words:", content_words[: 3]), print("\nLabel_words:", label_words[: 3])

# tokenize the field_names:
print("\n\nTokenizing the field_names ...")
field_sequences, field_dict, rev_field_dict, field_vocab_size = prepare_tokenizer(field_names)

print("Encoded field_sequences:", field_sequences[: 3])


#Last part is to tokenize the content and the label sequences together:
# note the length of the content_words:
content_split_point = len(content_words)

# attach them together
# transform the label_words to add <start> and <eos> tokens to all the sentences
for i in range(len(label_words)):
    label_words[i] = ['<start>'] + label_words[i] + ['<eos>']

unified_content_label_list = content_words + label_words

# tokenize the unified_content_and_label_words:
print("\n\nTokenizing the content and the label names ...")
unified_sequences, content_label_dict, rev_content_label_dict, content_label_vocab_size = prepare_tokenizer(unified_content_label_list, max_word_length = 20000)

print("Encoded content_label_sequences:", unified_sequences[: 3])

# obtain the content and label sequences by separating it from the unified_sequences
content_sequences = unified_sequences[: content_split_point]; label_sequences = unified_sequences[content_split_point: ]

# Finally, pickle all of it together:
pickle_obj = {
    # ''' Input structured data: '''
    
    # field_encodings and related data:
    'field_encodings': field_sequences,
    'field_dict': field_dict,
    'field_rev_dict': rev_field_dict,
    'field_vocab_size': field_vocab_size,
    
    # content encodings and related data:
    'content_encodings': content_sequences,
    
    # ''' Label summary sentences: '''
    
    # label encodings and related data:
    'label_encodings': label_sequences,
    
    # V union C related data:
    'content_union_label_dict': content_label_dict,
    'rev_content_union_label_dict': rev_content_label_dict,
    'content_label_vocab_size': content_label_vocab_size
}

# call the pickling function to perform the pickling:
print("\nPickling the processed data ...")
pickleIt(pickle_obj, "../Data/plug_and_play.pickle")
