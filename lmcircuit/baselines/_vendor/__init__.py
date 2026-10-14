"""Vendored slices of two published libraries, used unmodified in their algorithms (Apache License 2.0, each LICENSE
alongside). Every modified file says so in its first line (Apache-2.0 section 4b).

frugalgpt/        stanford-futuredata/FrugalGPT (https://github.com/stanford-futuredata/FrugalGPT), Apache-2.0.
                  llmchain.py and optimizer.py are src/FrugalGPT/{llmchain,optimizer}.py; utils.py holds the functions
                  of src/service/utils.py that LLMChain calls. Identical to the files at commit
                  2b23e6acbd4b39f71912816640dcc59e45aa4b5e (main, 2025-02-09; the three files last changed in
                  41b5bbfaa1126fdc10b51253a6cacd303719fc0b, 2023-06-22) apart from the import lines noted in them.
cascade_routing/  eth-sri/cascade-routing (https://github.com/eth-sri/cascade-routing), Apache-2.0. The files of
                  src/selection/ the classification setting uses (CascadeRouter, ClassificationQualityComputer,
                  ClassificationCostComputer, ConstantStrategy and their bases) at commit
                  1788b904a42aa586d555784be240b344868480d2 (v1.1, 2025-02-14), with the changes noted in
                  cascade_router.py, classification.py, lambda_strategy.py and statistics.py.

Both were checked file by file against those commits on 2026-10-06.
"""
