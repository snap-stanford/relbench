import pooch

__version__ = "0.1.1"

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
        "rel-stackex/db.zip": "deb00ccdf825e569b34935834444429cd1c0074b50226b12d616aab22d36242d",
        "rel-stackex/tasks/rel-stackex-engage.zip": "9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18",
        "rel-stackex/tasks/rel-stackex-votes.zip": "0dab5bebd76a95d689c8a3a62026c1c294a252c561fd940e8d9329d165d98a5a",
        "rel-amazon-books_5_core/db.zip": "2f6bd920bcfe08cbb7d47115f47f8d798a2ec1a034b6c2f3d8d9906e967454b4",
        "rel-amazon-books_5_core/tasks/rel-amazon-churn.zip": "d3890621b1576a9d5b6bc273cdd2ea2084aeaf9c8055c1421ded84be0c48dacb",
        "rel-amazon-books_5_core/tasks/rel-amazon-ltv.zip": "2e91be0ca5d9f591d8e33a40f70b97db346090a8bb9f3a94f49b147f0dc136be",
        "rel-trial/db.zip": "76093dae4365839cae4f949cc2c982c8c8ddf9886e309d84606b37208c8102da",
        "rel-math-stackex/db.zip": "00b193587f1ee0b39c77d2b561385f00fbc3b4ca0929f420e16ddd53115ce3ea",
        "rel-f1/db.zip": "e41ca0d69d54f16b408fe03b6c19b772ae701336cf84260ef5c84fca798a1422",
        "rel-hm/db.zip": "6ff6537f2fed885c5c8a94525364678dea206c57006de0edb4d76ca71c9c114e",
        "rel-avito/db.zip": "09fe913ece4f17f79ca0d2c1d25ed9f6f7e803fa4a08dcf520b7a0e73f34b1ed",
        "rel-event/db.zip": "141f4842600d091250c1f94e4c479c35e76d7ec3aef9155316f83d4828d85e5e",
        "rel-amazon/db.zip": "2fb5d1b6f0d8886374bc25b3a81becbe191ad30e524ac7fb998bab4c87010adc",
        "rel-stack/db.zip": "b703d141f86c210e9e6809807ec0bcf9b3e2fcd32a679835c9d71d8048b89188",
    },
)

# TODO: remove
new_name = {
    # datasets
    "rel-amazon": "rel-amazon",
    "rel-avito": "rel-avito",
    "rel-event": "rel-event",
    "rel-f1": "rel-f1",
    "rel-hm": "rel-hm",
    "rel-stackex": "rel-stack",
    "rel-trial": "rel-trial",
    # tasks
    ## rel-amazon
    "rel-amazon-churn": "user-churn",
    "rel-amazon-ltv": "user-ltv",
    "rel-amazon-product-churn": "item-churn",
    "rel-amazon-product-ltv": "item-ltv",
    "rel-amazon-rec-purchase": "user-item-purchase",
    "rel-amazon-rec-5-star": "user-item-rate",
    "rel-amazon-rec-detailed-review": "user-item-review",
    ## rel-avito
    "rel-avito-click": "user-clicks",
    "rel-avito-rec": "user-ad-click",
    ## rel-event
    "rel-event-attendence": "user-attendance",
    ## rel-f1
    "rel-f1-position": "driver-position",
    "rel-f1-dnf": "driver-dnf",
    "rel-f1-qualifying": "driver-top3",
    ## rel-hm
    "rel-hm-rec": "user-item-purchase",
    "rel-hm-churn": "user-churn",
    "rel-hm-sales": "item-sales",
    ## rel-stack
    "rel-stackex-engage": "user-engagement",
    "rel-stackex-votes": "post-votes",
    "rel-stackex-badges": "user-badge",
    "rel-stackex-comment-on-post": "user-post-comment",
    "rel-stackex-related-post": "post-post-related",
    ## rel-trial
    "rel-trial-outcome": "study-outcome",
    "rel-trial-adverse": "study-adverse",
    "rel-trial-withdrawal": "study-withdrawal",
    "rel-trial-site": "site-success",
    "rel-trial-sponsor-condition": "condition-sponsor-run",
    "rel-trial-sponsor-facility": "site-sponsor-run",
}
