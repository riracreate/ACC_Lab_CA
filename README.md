# LambdaWarm - ML-based proactive Provisioned Concurrency for AWS Lambda

Advanced Cloud Computing - Lab CA. Praneet Sharma (16010123237), Vidhan Aggarwal (16010123263)

## Files
| File | Runs on | Purpose |
|---|---|---|
| `lambda_function.py` | AWS Lambda | The function being scaled. Slow init (1.8 s) = cold start; reports whether it ran on provisioned concurrency. |
| `lambdawarm.py` | laptop | Everything else: data loader, features, LightGBM models, 5 policies, simulator, cost/energy model, plots, boto3 controller. |
| `run_experiments.py` | laptop | One command that produces every table and figure for the report (`results/`). |
| `live_demo.py` | laptop -> AWS | 8-10 minute real-AWS test: cold vs warm, wrong prediction, fail-safe. Deletes PC at the end. |
| `app.py` | laptop | Streamlit dashboard for the screencast. |

## Quick start
```bash
python3 -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt                         # macOS + LightGBM error about libomp?  brew install libomp
python run_experiments.py --synthetic                   # 10-second smoke test, no downloads
```
Real data: download the Azure Functions 2019 trace
(https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz, 143 MB),
then extract only what you need into `data/azure/`:
```bash
mkdir -p data/azure
tar -xf azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz -C data/azure \
    invocations_per_function_md.anon.d01.csv invocations_per_function_md.anon.d02.csv invocations_per_function_md.anon.d03.csv
python run_experiments.py --azure-dir data/azure        # first run ~1 min (scans CSVs, then cached)
streamlit run app.py
```
AWS (after creating the function, version and alias - see the guide):
```bash
aws configure                                           # paste the access key once; never put keys in code or on screen
python live_demo.py --function CloudWorkloadTarget --alias prod --region ap-south-1 --n 3
```

## Data licence
Azure Functions Trace 2019 is CC-BY. Cite: Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the
Serverless Workload at a Large Cloud Provider", USENIX ATC 2020.
