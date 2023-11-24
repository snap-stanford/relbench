import os
import random
import string
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from rtb.data import Database, Dataset, Table, Task
from rtb.tasks.amazon_reviews import CustomerChurnTask, CustomerLTVTask


def _generate_random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(length))
    return random_string


class FakeReviewsDataset(Dataset):
    def __init__(
        self, num_products: int = 30, num_customers: int = 100, num_reviews: int = 500
    ):
        db = self.process_db(num_products, num_customers, num_reviews)
        db.reindex_pkeys_and_fkeys()
        val_timestamp = db.min_timestamp + 0.8 * (db.max_timestamp - db.min_timestamp)
        test_timestamp = db.min_timestamp + 0.9 * (db.max_timestamp - db.min_timestamp)
        super().__init__(
            db=db,
            val_timestamp=val_timestamp,
            test_timestamp=test_timestamp,
            task_cls_dict={
                "customer_churn": CustomerChurnTask,
                "customer_ltv": CustomerLTVTask,
            },
        )

    def process_db(self, num_products, num_customers, num_reviews) -> Database:
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
                "review_time": pd.to_datetime(10 * np.arange(num_reviews), unit="D"),
                "rating": np.random.randint(1, 6, size=(num_reviews,)),
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
            }
        )
