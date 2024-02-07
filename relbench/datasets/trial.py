import os

import pandas as pd
import numpy as np
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.trial import OutcomeTask
from relbench.utils import unzip_processor


class TrialDataset(RelBenchDataset):
    name = "rel-trial"
    # 2 years gap
    val_timestamp = pd.Timestamp("2019-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")
    task_cls_list = [OutcomeTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-trial-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="455a544ca917e43175032ac21ad43395dae468f9853463f383bc94ddd3e5f83c",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "relbench-trial-raw")
        studies = pd.read_csv(os.path.join(path, "studies.txt"), sep = '|')
        outcomes = pd.read_csv(os.path.join(path, "outcomes.txt"), sep = '|')
        drop_withdrawals = pd.read_csv(os.path.join(path, "drop_withdrawals.txt"), sep = '|')
        designs = pd.read_csv(os.path.join(path, "designs.txt"), sep = '|')
        eligibilities = pd.read_csv(os.path.join(path, "eligibilities.txt"), sep = '|')
        interventions = pd.read_csv(os.path.join(path, "interventions.txt"), sep = '|')
        conditions = pd.read_csv(os.path.join(path, "conditions.txt"), sep = '|')
        reported_event_totals = pd.read_csv(os.path.join(path, "reported_event_totals.txt"), sep = '|')
        sponsors = pd.read_csv(os.path.join(path, "sponsors.txt"), sep = '|')
        facilities = pd.read_csv(os.path.join(path, "facilities.txt"), sep = '|')
        outcome_analyses = pd.read_csv(os.path.join(path, "outcome_analyses.txt"), sep = '|')
        detailed_descriptions = pd.read_csv(os.path.join(path, "detailed_descriptions.txt"), sep = '|')
        brief_summaries = pd.read_csv(os.path.join(path, "brief_summaries.txt"), sep = '|')
                
        ## just using trials with actual completion date 
        #print('studies', len(studies))
        studies = studies[studies.completion_date_type == 'Actual']
        ## there are 27 trials before 1975
        studies = studies[studies.start_date > '1975-01-01']
        studies = studies[studies.nct_id.notnull()]
        #print('studies actual', len(studies))
        nct_id_use = studies.nct_id.values
        
         ## get trial start and end date for later infer
        studies['start_date'] = pd.to_datetime(studies['start_date'])
        studies['completion_date'] = pd.to_datetime(studies['completion_date'])
        nct2start_date = dict(studies[['nct_id', 'start_date']].values)
        nct2end_date = dict(studies[['nct_id', 'completion_date']].values)
        
        ## too many columns in studies, keeping few interesting columns and remove temporal leakage columns
        studies = studies[['nct_id', 'start_date', 'target_duration', 'study_type',
       'acronym', 'baseline_population', 'brief_title', 'official_title', 'phase', 
        'enrollment', 'enrollment_type', 'source', 'limitations_and_caveats',
       'number_of_arms', 'number_of_groups', 'has_dmc',
       'is_fda_regulated_drug', 'is_fda_regulated_device',
       'is_unapproved_device', 'is_ppsd', 'is_us_export',
       'biospec_retention', 'biospec_description','source_class', 
        'baseline_type_units_analyzed','fdaaa801_violation','plan_to_share_ipd']]

        ## merge description/brief into main study table
        nct2descriptions = dict(detailed_descriptions[['nct_id', 'description']].values)
        nct2brief = dict(brief_summaries[['nct_id', 'description']].values)
        studies['detailed_descriptions'] = studies.nct_id.apply(lambda x: nct2descriptions[x] if x in nct2descriptions else np.nan)
        studies['brief_summaries'] = studies.nct_id.apply(lambda x: nct2brief[x] if x in nct2brief else np.nan)
        
        outcomes = outcomes[['id', 'nct_id', 'outcome_type', 'title', 'description','time_frame', 'population','units', 'units_analyzed',
       'dispersion_type', 'param_type']]

        reported_event_totals = reported_event_totals[['id', 'nct_id', 'event_type', 'classification', 'subjects_affected', 'subjects_at_risk']]

        drop_withdrawals.drop(columns=['result_group_id', 'ctgov_group_code', 'drop_withdraw_comment', 'reason_comment', 'count_units'], inplace = True)
        ## filter to nct_id with actual completion date
        #print('outcomes before filter', len(outcomes))
        #for df in [outcomes, outcome_analyses, drop_withdrawals, reported_event_totals, designs, eligibilities, interventions, conditions, facilities, sponsors]:
        
        outcomes = outcomes[outcomes.nct_id.isin(nct_id_use)]
        outcome_analyses = outcome_analyses[outcome_analyses.nct_id.isin(nct_id_use)]
        drop_withdrawals = drop_withdrawals[drop_withdrawals.nct_id.isin(nct_id_use)]
        reported_event_totals = reported_event_totals[reported_event_totals.nct_id.isin(nct_id_use)]
        designs = designs[designs.nct_id.isin(nct_id_use)]
        eligibilities = eligibilities[eligibilities.nct_id.isin(nct_id_use)]
        interventions = interventions[interventions.nct_id.isin(nct_id_use)]
        conditions = conditions[conditions.nct_id.isin(nct_id_use)]
        facilities = facilities[facilities.nct_id.isin(nct_id_use)]
        sponsors = sponsors[sponsors.nct_id.isin(nct_id_use)]
            
        #print('outcomes after filter', len(outcomes))
        ## infer time stamps
        ## tables that is available after trial ends
        for df in [outcomes, outcome_analyses, drop_withdrawals, reported_event_totals]:
            df['inferred_date'] = df.nct_id.apply(lambda x: nct2end_date[x])

        ## tables that is available as trial starts
        for df in [designs, eligibilities, interventions, conditions, facilities, sponsors]:
            df['inferred_date'] = df.nct_id.apply(lambda x: nct2start_date[x])
            
        tables = {}

        tables["studies"] = Table(
            df=studies,
            fkey_col_to_pkey_table={},
            pkey_col="nct_id",
            time_col="start_date",
        )

        tables["outcomes"] = Table(
            df=outcomes,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["outcome_analyses"] = Table(
            df=outcome_analyses,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
                "outcome_id": "outcomes"
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["drop_withdrawals"] = Table(
            df=drop_withdrawals,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["reported_event_totals"] = Table(
            df=reported_event_totals,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["designs"] = Table(
            df=designs,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["eligibilities"] = Table(
            df=eligibilities,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["interventions"] = Table(
            df=interventions,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["conditions"] = Table(
            df=conditions,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["facilities"] = Table(
            df=facilities,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )

        tables["sponsors"] = Table(
            df=sponsors,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="inferred_date",
        )
        return Database(tables)
