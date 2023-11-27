import pooch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "relbench",
]

# TODO: use the versioning feature of pooch
_pooch = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry={
        # extremely small dataset only used for testing download functionality
        "rel-amazon-fashion_5_core/db.zip": "27e08bc808438e8619560c54d0a4a7a11e965b90b8c70ef3a0928b44a46ad028",
        "rel-amazon-fashion_5_core/tasks/rel-amazon-churn.zip": "d98f2240aefa0f175dab2fce4a48a1cc595be584d4960cd9eb750d012326117d",
        "rel-amazon-fashion_5_core/tasks/rel-amazon-ltv.zip": "bd2b7b798efad2838a3701def8386dba816b45ef277a8e831052b79f5448aed8",
        "rel-stackex/db.zip": "dfb84faa4918c6c4ecac791a69a30a477a7bee097d7295d48c78ceb8f59c997c",
        "rel-stackex/tasks/rel-stackex-engage.zip": "9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18",
        "rel-stackex/tasks/rel-stackex-votes.zip": "0dab5bebd76a95d689c8a3a62026c1c294a252c561fd940e8d9329d165d98a5a",
        "rel-amazon-books_5_core/db.zip": "2f6bd920bcfe08cbb7d47115f47f8d798a2ec1a034b6c2f3d8d9906e967454b4",
        "rel-amazon-books_5_core/tasks/rel-amazon-churn.zip": "d3890621b1576a9d5b6bc273cdd2ea2084aeaf9c8055c1421ded84be0c48dacb",
        "rel-amazon-books_5_core/tasks/rel-amazon-ltv.zip": "2e91be0ca5d9f591d8e33a40f70b97db346090a8bb9f3a94f49b147f0dc136be",
    },
)
