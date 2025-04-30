eval "$(conda shell.bash hook)"
conda activate matsimAIenv

cd ..

python3 -m matsimAI.scripts.analysis \
"./cluster_results/results/0428162326_nclusters_10_utahnetwork" \
"./scenarios/cluster_scenarios/utah_10c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_10c/utahcounts.xml" \
"./outputs/output_10c" \

python3 -m matsimAI.scripts.analysis \
"./cluster_results/results/0428164930_nclusters_50_utahnetwork" \
"./scenarios/cluster_scenarios/utah_50c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_50c/utahcounts.xml" \
"./outputs/output_50c" \

python3 -m matsimAI.scripts.analysis \
"./cluster_results/results/0428172102_nclusters_100_utahnetwork" \
"./scenarios/cluster_scenarios/utah_100c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_100c/utahcounts.xml" \
"./outputs/output_100c" \

python3 -m matsimAI.scripts.analysis \
"./cluster_results/results/0428175625_nclusters_200_utahnetwork" \
"./scenarios/cluster_scenarios/utah_200c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_200c/utahcounts.xml" \
"./outputs/output_200c" \

python3 -m matsimAI.scripts.analysis \
"./cluster_results/results/0429160536_nclusters_500_utahnetwork" \
"./scenarios/cluster_scenarios/utah_500c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_500c/utahcounts.xml" \
"./outputs/output_500c" \