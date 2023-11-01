import rtb


def churn(db: rtb.data.Database, time_frame: str = "1W") -> rtb.data.Task:
    r"""Create Task object for churn."""

    raise NotImplementedError


def ltv(db: rtb.data.Database, time_frame: str = "1W") -> rtb.data.Task:
    r"""Create Task object for LTV.

    LTV (life-time value) for a customer is the sum of prices of products that
    the user reviews in the time_frame.

    See:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    for time_frame strings.

    """

    # select the relevant columns
    product = db.tables["product"].df[["product_id", "price"]]
    review = db.tables["review"].df[["time_stamp", "customer_id", "product_id"]]

    # join the tables
    df = review.merge(product, on="product_id")

    # remove the product_id column
    df = df.drop(columns=["product_id"])

    # pandas rolling window works for datetime columns
    assert is_datetime64_any_dtype(df.dtypes["time_stamp"])

    # group by customer_id and sum the prices in each rolling window

    # TODO: as is, this puts the value with the last time_stamp in the window
    # we want the first time_stamp in the window, but didn't find a kwarg for this
    df = (
        df.groupby("customer_id")
        .rolling(
            window=time_frame,
            on="time_stamp",
            closed="right",  # skip the first time_stamp in the window
        )
        .sum()
    )

    # this does not support strided rolling windows, but we don't need that
    # to get fewer datapoints just subsample the task table

    # df columns are: time_stamp, customer_id, price

    return rtb.data.task.Task(
        table=rtb.data.table.Table(
            df=df,
            feat_cols={"price": rtb.data.table.SemanticType.NUMERICAL},
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="time_stamp",
        ),
        label_col="price",
        task_type=rtb.data.task.TaskType.REGRESSION,
        metrics=["mse", "smape"],
    )


class ProductDataset(rtb.data.Dataset):
    name = "mtb-product"
    task_fns = {"churn": churn, "ltv": ltv}

    def download(self) -> None:
        r"""Download the Amazon dataset raw files from the AWS server."""

        raise NotImplementedError

    def process_db(self) -> rtb.data.Database:
        r"""Process the raw files into a database."""

        raise NotImplementedError
