import rtb


def churn(db: rtb.data.Database, time_frame: str = "1W") -> rtb.data.Task:
    r"""Create Task object for churn."""

    raise NotImplementedError


def ltv(
    db: rtb.data.Database, start_time_stamp: int, end_time_stamp: int, time_frame: int
) -> rtb.data.Task:
    r"""Create Task object for LTV.

    LTV (life-time value) for a customer is the sum of prices of products that
    the user reviews in the time_frame.

    Simply groups events into time windows of size time_frame. Windows do not
    overlap, so this is not the same as a rolling window. Rolling window can
    also be implemented, but we don't need it for now.
    """

    # select the relevant columns
    product = db.tables["product"].df[["product_id", "price"]]
    review = db.tables["review"].df[["time_stamp", "customer_id", "product_id"]]

    # join the tables
    df = review.merge(product, on="product_id")

    # filter out events that are before begin_time_stamp or after end_time_stamp
    df = df.query(
        f"(time_stamp >= {start_time_stamp}) & (time_stamp < {end_time_stamp})"
    )

    # compute the left time stamp for each event
    df["left_time_stamp"] = (
        (df["time_stamp"] - start_time_stamp) // time_frame * time_frame
    )

    # remove left_time_stamp > end_time_stamp - time_frame because it's time window
    # got truncated
    df = df.query(f"left_time_stamp <= {end_time_stamp - time_frame}")

    # remove unnecessary columns
    df = df.drop(columns=["product_id", "time_stamp"])

    df = df.groupby(["customer_id", "left_time_stamp"]).sum()

    # columns of df: customer_id, left_time_stamp, price

    return rtb.data.task.Task(
        table=rtb.data.table.Table(
            df=df,
            feat_cols={"price": rtb.data.table.SemanticType.NUMERICAL},
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="left_time_stamp",
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
