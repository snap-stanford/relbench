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
        "amazon_reviews-fashion_5_core/db.zip": "53976c20468e5905cdbcf6ff1621f052febaf76b40c16a2e8816d9dee9a51e82",
        "amazon_reviews-fashion_5_core/tasks/customer_churn.zip": "04d68586a81c462f85ba3292f4df367b42ef06d41b090c9077995eccdb74a68f",
        "amazon_reviews-fashion_5_core/tasks/customer_ltv.zip": "7366598d25fddd8c995fdff10e3c56551966250b3161af996e6b42f546c5b931",
        "stack_exchange/db.zip": "b46e996d6659819fb4fd54cac5cba71fb3bfd8a5b2705fc44580603c0d2fe0e9",
        "stack_exchange/tasks/question_popularity.zip": "de4bacb7c31d23733aedf99999fe859977c3d1f5b54d0a7af644d1b2a4c60f41",
        "stack_exchange/tasks/user_contribution.zip": "c9c6a7554dd788c4272e7035bbb6391428b20fc1e10e8065cad554a6c75df5e1",
        
    },
)
