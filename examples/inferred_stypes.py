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
            # "LastEditorUserId": stype.numerical,
            # Uninformative text column
            # "LastEditorDisplayName": stype.text_embedded,
            "Title": stype.text_embedded,
            "Tags": stype.text_embedded,
        },
        "users": {
            "Id": stype.numerical,
            "AccountId": stype.numerical,
            "CreationDate": stype.timestamp,
            # Uninformative text column
            # "DisplayName": stype.text_embedded,
            # "Location": stype.text_embedded,
            "AboutMe": stype.text_embedded,
            # Uninformative text column
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
            # Uninformative text column
            # "UserDisplayName": stype.text_embedded,
            # "ContentLicense": stype.text_embedded,
        },
        "badges": {
            "Id": stype.numerical,
            "UserId": stype.numerical,
            "Class": stype.categorical,
            # Uninformative text column
            # "Name": stype.text_embedded,
            "Date": stype.timestamp,
            "TagBased": stype.categorical,
        },
        "postHistory": {
            "Id": stype.numerical,
            "PostId": stype.numerical,
            "UserId": stype.numerical,
            "PostHistoryTypeId": stype.numerical,
            # Uninformative text column
            # "UserDisplayName": stype.text_embedded,
            "ContentLicense": stype.categorical,
            # Uninformative text column
            # "RevisionGUID": stype.text_embedded,
            "Text": stype.text_embedded,
            # Uninformative text column
            # "Comment": stype.text_embedded,
            "CreationDate": stype.timestamp,
        },
    },
    'rel-f1': {
        'races': {
            'raceId': stype.numerical,
            'year': stype.numerical,
            'round': stype.numerical,
            'circuitId': stype.numerical,
            'name': stype.text_embedded,
            'date': stype.timestamp,
            'time': stype.timestamp,
        },
        'circuits': {
            'circuitId': stype.numerical,
            'circuitRef': stype.text_embedded,
            'name': stype.text_embedded,
            'location': stype.text_embedded,
            'country': stype.categorical,
            'lat': stype.numerical,
            'lng': stype.numerical,
            'alt': stype.numerical,
        },
        'drivers': {
            'driverId': stype.numerical,
            'driverRef': stype.text_embedded,
            'code': stype.text_embedded,
            'forename': stype.text_embedded,
            'surname': stype.text_embedded,
            'dob': stype.timestamp,
            'nationality': stype.categorical,
        },
        'results': {
            'resultId': stype.numerical,
            'raceId': stype.numerical,
            'driverId': stype.numerical,
            'number': stype.numerical,
            'grid': stype.numerical,
            'position': stype.numerical,
            'positionOrder': stype.numerical,
            'points': stype.numerical,
            'laps': stype.numerical,
            'time': stype.timestamp,
            'milliseconds': stype.numerical,
            'fastestLap': stype.numerical,
            'rank': stype.numerical,
            'fastestLapTime': stype.timestamp,
            'fastestLapSpeed': stype.numerical,
            'date': stype.timestamp,
        },
        'standings': {
            'driverStandingsId': stype.numerical,
            'raceId': stype.numerical,
            'driverId': stype.numerical,
            'points': stype.numerical,
            'position': stype.numerical,
            'wins': stype.numerical,
            'date': stype.timestamp,
        },
        'constructors': {
            'constructorId': stype.numerical,
            'constructorRef': stype.text_embedded,
            'name': stype.text_embedded,
            'nationality': stype.categorical,
        },
        'constructor_results': {
            'constructorResultsId': stype.numerical,
            'raceId': stype.numerical,
            'constructorId': stype.numerical,
            'points': stype.numerical,
            'status': stype.text_embedded,
            'date': stype.timestamp,
        },   
        'constructor_standings': {
            'constructorStandingsId': stype.numerical,
            'raceId': stype.numerical,
            'constructorId': stype.numerical,
            'points': stype.numerical,
            'position': stype.numerical,
            'wins': stype.numerical,
            'date': stype.timestamp,
        },
    }
}
