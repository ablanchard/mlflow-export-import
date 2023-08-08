""" 
Exports an experiment to a directory.
"""

import mlflow

from mlflow_export_import.common.iterators import SearchRunsIterator
import time


client = mlflow.tracking.MlflowClient()
all_run_ids = []
start_time = time.time()
iterator = SearchRunsIterator(client, "103062", max_results=1)
for j,run in enumerate(iterator):
    print(f"doing run {j}, {iterator.paged_list.token}, {iterator.max_results}")
    all_run_ids.append(run.info.run_id)

print(f"done {len(all_run_ids)} runs in {time.time() - start_time}")