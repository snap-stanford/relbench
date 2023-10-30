import rtb

# should move it to a better place
WEEK = 7 * 24 * 60 * 60  # we use seconds as the unit of time everywhere


def churn(db: rtb.data.Database, time_frame: int = WEEK) -> rtb.data.Task:
    r"""Create Task object for churn."""

    raise NotImplementedError


def ltv(db: rtb.data.Database, time_frame: int = WEEK) -> rtb.data.Task:
    r"""Create Task object for LTV."""

    raise NotImplementedError


class ProductDataset(rtb.data.Dataset):
    name = "mtb-product"
    task_fns = {"churn": churn, "ltv": ltv}

    def download(self) -> None:
        r"""Download the Amazon dataset raw files from the AWS server."""

        raise NotImplementedError

    def process_db(self) -> rtb.data.Database:
        r"""Process the raw files into a database."""

        raise NotImplementedError
