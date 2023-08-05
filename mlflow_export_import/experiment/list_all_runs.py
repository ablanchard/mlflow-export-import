""" 
Exports an experiment to a directory.
"""

import mlflow

from mlflow_export_import.common.iterators import SearchRunsIterator
import time


client = mlflow.tracking.MlflowClient()
all_run_ids = []
start_time = time.time()
iterator = SearchRunsIterator(client, "1409965")
for j,run in enumerate(iterator):
    print(f"doing run {j}, {iterator.paged_list.token}")
    all_run_ids.append(run.info.run_id)

print(f"done {len(all_run_ids)} runs in {start_time - time.time()}")