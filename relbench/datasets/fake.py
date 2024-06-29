import random
import string

import numpy as np
import pandas as pd

from relbench.data import Database, Dataset, Table


def _generate_random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(length))
    return random_string


class FakeDataset(Dataset):
    def __init__(
        self,
        num_products: int = 30,
        num_customers: int = 100,
        num_reviews: int = 600,
        num_relations: int = 20,
    ):
        self.num_products = num_products
        self.num_customers = num_customers
        self.num_reviews = num_reviews
        self.num_relations = num_relations

        min_timestamp = pd.Timestamp(0, unit="D")
        max_timestamp = pd.Timestamp(2 * (num_reviews - 1), unit="D")
        self.val_timestamp = min_timestamp + 0.8 * (max_timestamp - min_timestamp)
        self.test_timestamp = min_timestamp + 0.9 * (max_timestamp - min_timestamp)
        self.max_eval_time_frames = 1
        super().__init__()

    def make_db(self) -> Database:
        num_products = self.num_products
        num_customers = self.num_customers
        num_reviews = self.num_reviews
        num_relations = self.num_relations
        product_df = pd.DataFrame(
            {
                "product_id": [f"product_id_{i}" for i in range(num_products)],
                "category": [None, [], ["toy", "health"]] * (num_products // 3),
                "title": [_generate_random_string(5, 15) for _ in range(num_products)],
                "price": np.random.rand(num_products) * 10,
            }
        )
        customer_df = pd.DataFrame(
            {
                "customer_id": [f"customer_id_{i}" for i in range(num_customers)],
                "age": np.random.randint(10, 50, size=(num_customers,)),
                "gender": ["male", "female"] * (num_customers // 2),
            }
        )
        # Add some dangling foreign keys:
        review_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_reviews)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_reviews)
                ],
                "review_time": pd.to_datetime(2 * np.arange(num_reviews), unit="D"),
                "rating": np.random.randint(1, 6, size=(num_reviews,)),
            }
        )
        relations_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_relations)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_relations)
                ],
            }
        )

        return Database(
            table_dict={
                "product": Table(
                    df=product_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="product_id",
                ),
                "customer": Table(
                    df=customer_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
                ),
                "review": Table(
                    df=review_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                    time_col="review_time",
                ),
                "relations": Table(
                    df=relations_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                ),
            }
        )
