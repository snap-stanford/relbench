# this file contains the informative columns for each dataset

# TODO (joshrob) move to dataset definition datasets/amazon.py etc.

dataset_to_informative_text_cols = {'rel-stackex': {
                                                   "postHistory": ["Text"],
                                                   "users": ["AboutMe"],
                                                   "posts": ["Body", "Title", "Tags"],
                                                   "comments": ["Text"],
                                                   },
                                    'rel-amazon': {
                                                      'product': ['brand', 'title', 'description'],
                                                      'customer': ['customer_name'],
                                                      'review': ['review_text', 'summary'],
                                                   }
                                    }


