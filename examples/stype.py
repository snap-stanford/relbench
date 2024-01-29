from torch_frame import stype

# TODO (joshrob) move to dataset definition datasets/amazon.py etc.

stype_dict = {'rel-amazon':
                {'product': {'product_id': stype.numerical,
                             'brand': stype.text_embedded,
                             'title': stype.text_embedded,
                             'description': stype.text_embedded,
                             'price': stype.numerical,
                             'category': stype.multicategorical},
                'customer': {'customer_id': stype.numerical,
                             'customer_name': stype.text_embedded},
                'review':   {'review_text': stype.text_embedded,
                             'summary': stype.text_embedded,
                             'review_time': stype.timestamp,
                             'rating': stype.numerical,
                             'verified': stype.categorical,
                             'customer_id': stype.numerical,
                             'product_id': stype.numerical}
                }
             }       
                 


