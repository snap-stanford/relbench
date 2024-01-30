from torch_frame import stype

# TODO (joshrob) move to dataset definition datasets/amazon.py etc.

dataset2inferred_stypes = {
    "rel-amazon": {
        "product": {
            "product_id": stype.numerical,
            "brand": stype.text_embedded,
            "title": stype.text_embedded,
            "description": stype.text_embedded,
            "price": stype.numerical,
            "category": stype.multicategorical,
        },
        "customer": {
            "customer_id": stype.numerical,
            "customer_name": stype.text_embedded,
        },
        "review": {
            "review_text": stype.text_embedded,
            "summary": stype.text_embedded,
            "review_time": stype.timestamp,
            "rating": stype.numerical,
            "verified": stype.categorical,
            "customer_id": stype.numerical,
            "product_id": stype.numerical,
        },
    },
    "rel-stackex": {
        "postLinks": {
            "Id": stype.numerical,
            "RelatedPostId": stype.numerical,
            "PostId": stype.numerical,
            "LinkTypeId": stype.numerical,
            "CreationDate": stype.timestamp,
        },
        "posts": {
            "Id": stype.numerical,
            "PostTypeId": stype.numerical,
            "AcceptedAnswerId": stype.numerical,
            "ParentId": stype.numerical,
            "CreationDate": stype.timestamp,
            "Body": stype.text_embedded,
            "OwnerUserId": stype.numerical,
            "LastEditorUserId": stype.numerical,
            # Uninformative
            # "LastEditorDisplayName": stype.text_embedded,
            "Title": stype.text_embedded,
            "Tags": stype.text_embedded,
        },
        "users": {
            "Id": stype.numerical,
            "AccountId": stype.numerical,
            "CreationDate": stype.timestamp,
            # Uninformative
            # "DisplayName": stype.text_embedded,
            # "Location": stype.text_embedded,
            "AboutMe": stype.text_embedded,
            # Uninformative
            # "WebsiteUrl": stype.text_embedded,
        },
        "votes": {
            "Id": stype.numerical,
            "PostId": stype.numerical,
            "VoteTypeId": stype.numerical,
            "UserId": stype.numerical,
            "CreationDate": stype.timestamp,
        },
        "comments": {
            "Id": stype.numerical,
            "PostId": stype.numerical,
            "Text": stype.text_embedded,
            "CreationDate": stype.timestamp,
            "UserId": stype.numerical,
            # Uninformative
            # "UserDisplayName": stype.text_embedded,
            # "ContentLicense": stype.text_embedded,
        },
        "badges": {
            "Id": stype.numerical,
            "UserId": stype.numerical,
            "Class": stype.categorical,
            # Uninformative
            # "Name": stype.text_embedded,
            "Date": stype.timestamp,
            "TagBased": stype.categorical,
        },
        "postHistory": {
            "Id": stype.numerical,
            "PostId": stype.numerical,
            "UserId": stype.numerical,
            "PostHistoryTypeId": stype.numerical,
            # Uninformative
            # "UserDisplayName": stype.text_embedded,
            "ContentLicense": stype.categorical,
            # Uninformative
            # "RevisionGUID": stype.text_embedded,
            "Text": stype.text_embedded,
            # Uninformative
            # "Comment": stype.text_embedded,
            "CreationDate": stype.timestamp,
        },
    },
}
