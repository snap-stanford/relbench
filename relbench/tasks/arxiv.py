import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    mae,
    r2,
    rmse,
    roc_auc,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


class AuthorCategoryTask(EntityTask):
    r"""Predict the primary research category in which an author will publish papers in the next year."""

    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "Author_ID"
    entity_table = "authors"
    time_col = "date"
    target_col = "primary_category"
    timedelta = pd.Timedelta(days=365)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        author = db.table_dict["authors"].df
        paperAuthors = db.table_dict["paperAuthors"].df
        papers = db.table_dict["papers"].df
        paperCategories = db.table_dict["paperCategories"].df
        categories = db.table_dict["categories"].df

        df = duckdb.sql(
            f"""
            WITH author_pubs AS (
                SELECT
                    t.timestamp AS date,
                    pa.Author_ID,
                    p.Primary_Category_ID
                FROM timestamp_df t
                JOIN paperAuthors pa 
                ON pa.Submission_Date > t.timestamp 
                AND pa.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                JOIN papers p 
                ON pa.Paper_ID = p.Paper_ID
            ),
            pub_counts AS (
                SELECT
                    date,
                    Author_ID,
                    Primary_Category_ID,
                    COUNT(*) AS cnt
                FROM author_pubs
                GROUP BY date, Author_ID, Primary_Category_ID
            ),
            ranked AS (
                SELECT
                    date,
                    Author_ID,
                    Primary_Category_ID,
                    ROW_NUMBER() OVER (PARTITION BY date, Author_ID ORDER BY cnt DESC) AS rn
                FROM pub_counts
            )
            SELECT
                r.date,
                r.Author_ID,
                c.Category AS primary_category
            FROM ranked r
            LEFT JOIN categories c ON r.Primary_Category_ID = c.Category_ID
            WHERE r.rn = 1
            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class PaperCitationTask(EntityTask):
    r"""Predict how many citations a paper will receive in the next year."""

    task_type = TaskType.REGRESSION
    entity_col = "Paper_ID"
    entity_table = "papers"
    time_col = "Submission_Date"
    target_col = "citation_count"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        papers = db.table_dict["papers"].df
        citations = db.table_dict["citations"].df

        df = duckdb.sql(
            f"""
            WITH paper_citations AS (
                SELECT
                    t.timestamp AS date,
                    p.Paper_ID,
                    COUNT(c.References_Paper_ID) AS citation_count
                FROM timestamp_df t
                JOIN papers p 
                    ON p.Submission_Date <= t.timestamp
                LEFT JOIN citations c 
                    ON c.References_Paper_ID = p.Paper_ID
                    AND c.Submission_Date > t.timestamp
                    AND c.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp, p.Paper_ID
            )
            SELECT date, Paper_ID, citation_count FROM paper_citations;
            """
        ).df()
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class AuthorCitationTask(EntityTask):
    r"""Predict the total number of citations an author will receive (including both published and upcoming papers) in the next year."""

    task_type = TaskType.REGRESSION
    entity_col = "Author_ID"
    entity_table = "authors"
    time_col = "date"
    target_col = "citation_count"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        paperAuthors = db.table_dict["paperAuthors"].df
        citations = db.table_dict["citations"].df

        df = duckdb.sql(
            f"""
            WITH author_citations AS (
                SELECT
                    t.timestamp AS date,
                    pa.Author_ID,
                    COUNT(c.References_Paper_ID) AS citation_count
                FROM timestamp_df t
                CROSS JOIN paperAuthors pa
                LEFT JOIN citations c 
                    ON c.References_Paper_ID = pa.Paper_ID
                    AND c.Submission_Date > t.timestamp
                    AND c.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp, pa.Author_ID
            )
            SELECT date, Author_ID, citation_count FROM author_citations;
            """
        ).df()
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class AuthorPublicationTask(EntityTask):
    r"""Predict how many papers an author will publish in the next year."""

    task_type = TaskType.REGRESSION
    entity_col = "Author_ID"
    entity_table = "authors"
    time_col = "date"
    target_col = "publication_count"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        paperAuthors = db.table_dict["paperAuthors"].df

        df = duckdb.sql(
            f"""
            WITH author_pubs AS (
                SELECT
                    t.timestamp AS date,
                    pa.Author_ID,
                    COUNT(pa.Paper_ID) AS publication_count
                FROM timestamp_df t
                JOIN paperAuthors pa 
                    ON pa.Submission_Date > t.timestamp
                    AND pa.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp, pa.Author_ID
            )
            SELECT date, Author_ID, publication_count FROM author_pubs;
            """
        ).df()
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class AuthorCollaborationTask(EntityTask):
    r"""Predict the list of authors an author will collaborate with in the next year."""

    task_type = TaskType.LINK_PREDICTION
    entity_col = "Author_ID"
    entity_table = "authors"
    time_col = "date"
    target_col = "collaborators"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_map, link_prediction_precision, link_prediction_recall]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        paperAuthors = db.table_dict["paperAuthors"].df

        df = duckdb.sql(
            f"""
            WITH collaboration AS (
                SELECT
                    t.timestamp AS date,
                    pa1.Author_ID,
                    pa2.Author_ID AS collaborator
                FROM timestamp_df t
                JOIN paperAuthors pa1 
                  ON pa1.Submission_Date > t.timestamp 
                  AND pa1.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                JOIN paperAuthors pa2 
                  ON pa1.Paper_ID = pa2.Paper_ID
                  AND pa1.Author_ID <> pa2.Author_ID
            )
            SELECT 
                date, 
                Author_ID, 
                array_agg(DISTINCT collaborator) AS collaborators
            FROM collaboration
            GROUP BY date, Author_ID
            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class CoCitationTask(EntityTask):
    r"""Predict which other papers will be cited together with a given paper in the next year."""

    task_type = TaskType.LINK_PREDICTION
    entity_col = "Paper_ID"
    entity_table = "papers"
    time_col = "date"
    target_col = "co_cited"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_map, link_prediction_precision, link_prediction_recall]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        papers = db.table_dict["papers"].df
        citations = db.table_dict["citations"].df

        df = duckdb.sql(
            f"""
            WITH paper_co_citations AS (
                SELECT
                    t.timestamp AS date,
                    p.Paper_ID,
                    c2.References_Paper_ID AS co_cited_paper
                FROM timestamp_df t
                JOIN papers p 
                  ON p.Submission_Date <= t.timestamp
                JOIN citations c1 
                  ON c1.References_Paper_ID = p.Paper_ID
                  AND c1.Submission_Date > t.timestamp 
                  AND c1.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
                JOIN citations c2 
                  ON c1.Paper_ID = c2.Paper_ID
                  AND c2.References_Paper_ID <> p.Paper_ID
                  AND c2.Submission_Date > t.timestamp 
                  AND c2.Submission_Date <= t.timestamp + INTERVAL '{self.timedelta}'
            )
            SELECT 
                date, 
                Paper_ID, 
                array_agg(DISTINCT co_cited_paper) AS co_cited
            FROM paper_co_citations
            GROUP BY date, Paper_ID
            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


if __name__ == "__main__":
    from relbench.datasets.arxiv import ArxivDataset

    ds = ArxivDataset()
    db = ds.make_db()
    task_table = CoCitationTask(dataset=ds).make_table(
        db, pd.date_range("2020-01-01", "2020-12-31", freq="1M")
    )
    breakpoint()
