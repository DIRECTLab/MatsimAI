eval "$(conda shell.bash hook)"
conda activate matsimenv

cd ..

python3 -m matsimAI.run_gradient_flow_matching \
"./cluster_results/results" \
"./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
"./scenarios/utah_flow_scenario_example/utahcounts.xml" \
"--num_clusters" "10" \
"--training_steps" "10_000" \
"--log_interval" "1_000" \
"--save_interval" "-1" \
"--best_plans_save_path" "./scenarios/cluster_scenarios/utah_10c"

# python3 -m matsimAI.run_gradient_flow_matching \
# "./cluster_results/results" \
# "./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
# "./scenarios/utah_flow_scenario_example/utahcounts.xml" \
# "--num_clusters" "50" \
# "--training_steps" "1_000_000" \
# "--log_interval" "1_000" \
# "--save_interval" "-1" \

# python3 -m matsimAI.run_gradient_flow_matching \
# "./cluster_results/results" \
# "./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
# "./scenarios/utah_flow_scenario_example/utahcounts.xml" \
# "--num_clusters" "100" \
# "--training_steps" "1_000_000" \
# "--log_interval" "1_000" \
# "--save_interval" "-1" \

# python3 -m matsimAI.run_gradient_flow_matching \
# "./cluster_results/results" \
# "./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
# "./scenarios/utah_flow_scenario_example/utahcounts.xml" \
# "--num_clusters" "200" \
# "--training_steps" "1_000_000" \
# "--log_interval" "1_000" \
# "--save_interval" "-1" \

# python3 -m matsimAI.run_gradient_flow_matching \
# "./cluster_results/results" \
# "./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
# "./scenarios/utah_flow_scenario_example/utahcounts.xml" \
# "--num_clusters" "500" \
# "--training_steps" "1_000_000" \
# "--log_interval" "1_000" \
# "--save_interval" "-1" \