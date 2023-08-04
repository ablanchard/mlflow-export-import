""" 
Exports an experiment to a directory.
"""

import os
import click
import mlflow

from mlflow_export_import.common.click_options import *
from mlflow_export_import.common import mlflow_utils
from mlflow_export_import.common.iterators import SearchRunsIterator
from mlflow_export_import.common import io_utils
from mlflow_export_import.run.export_run import RunExporter
from mlflow_export_import.common import utils
from datetime import datetime
import time


client = mlflow.tracking.MlflowClient()
all_run_ids = []
start_time = time.time()
for j,run in enumerate(SearchRunsIterator(client, "1409965")):
    print(f"doing run {j}")
    all_run_ids.append(run.info.run_id)

print(f"done {len(all_run_ids)} runs in {start_time - time.time()}")