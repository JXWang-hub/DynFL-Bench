import csv, glob, os, json
out = {}
for f in glob.glob("results/telemetry_*.csv"):
    name = os.path.basename(f)[len("telemetry_"):-4]
    accs = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            accs.append(round(float(row["global_acc"]), 4))
    out[name] = accs
print(json.dumps(out))
