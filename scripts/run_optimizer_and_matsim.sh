eval "$(conda shell.bash hook)"
conda activate matsimAIenv

cd ..

python3 -m matsimAI.run_gradient_flow_matching \
"./cluster_results/results" \
"./scenarios/cluster_scenarios/utah_10c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_10c/utahcounts.xml" \
"--num_clusters" "10" \
"--training_steps" "1_000_000" \
"--log_interval" "1_000" \
"--save_interval" "-1" \
"--best_plans_save_path" "./scenarios/cluster_scenarios/utah_10c"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/cluster_scenarios/utah_10c/utahconfig.xml"

python3 -m matsimAI.run_gradient_flow_matching \
"./cluster_results/results" \
"./scenarios/cluster_scenarios/utah_50c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_50c/utahcounts.xml" \
"--num_clusters" "50" \
"--training_steps" "1_000_000" \
"--log_interval" "1_000" \
"--save_interval" "-1" \
"--best_plans_save_path" "./scenarios/cluster_scenarios/utah_50c"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/cluster_scenarios/utah_50c/utahconfig.xml"

python3 -m matsimAI.run_gradient_flow_matching \
"./cluster_results/results" \
"./scenarios/cluster_scenarios/utah_100c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_100c/utahcounts.xml" \
"--num_clusters" "100" \
"--training_steps" "1_000_000" \
"--log_interval" "1_000" \
"--save_interval" "-1" \
"--best_plans_save_path" "./scenarios/cluster_scenarios/utah_100c"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/cluster_scenarios/utah_100c/utahconfig.xml"

python3 -m matsimAI.run_gradient_flow_matching \
"./cluster_results/results" \
"./scenarios/cluster_scenarios/utah_200c/utahnetwork.xml" \
"./scenarios/cluster_scenarios/utah_200c/utahcounts.xml" \
"--num_clusters" "200" \
"--training_steps" "1_000_000" \
"--log_interval" "1_000" \
"--save_interval" "-1" \
"--best_plans_save_path" "./scenarios/cluster_scenarios/utah_200c"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/cluster_scenarios/utah_200c/utahconfig.xml"