import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb   

from relbench.data import Database, RelBenchTask, Table
from relbench.data.task import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window



class PointsTask(RelBenchTask):
    r"""Predict the finishing position of each driver in a race."""
    name = "rel-f1-points"
    task_type = TaskType.REGRESSION #TaskType.BINARY_CLASSIFICATION 
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "points"
    timedelta = pd.Timedelta(days=180)
    metrics = [mae, rmse] #[average_precision, accuracy, f1, roc_auc]]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for results_position_next_race."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        
        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    sum(re.points) points
                FROM 
                    timestamp_df t
                LEFT JOIN 
                    results re
                ON 
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN 
                    drivers dri
                ON 
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId
                
            ;
            """
        ).df()

        # make into binary classification task
        #df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 10. else 0)


        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


