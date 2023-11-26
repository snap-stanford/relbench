import pooch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "rtb",
]

# TODO: use the versioning feature of pooch
_pooch = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry={
        "amazon_reviews-fashion_5_core/db.zip": "53976c20468e5905cdbcf6ff1621f052febaf76b40c16a2e8816d9dee9a51e82",
        "amazon_reviews-fashion_5_core/tasks/customer_churn.zip": "04d68586a81c462f85ba3292f4df367b42ef06d41b090c9077995eccdb74a68f",
        "amazon_reviews-fashion_5_core/tasks/customer_ltv.zip": "7366598d25fddd8c995fdff10e3c56551966250b3161af996e6b42f546c5b931",
    },
)
