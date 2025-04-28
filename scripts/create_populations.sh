eval "$(conda shell.bash hook)"
conda activate matsimenv

cd ..

python3 -m matsimAI.scripts.create_population \
"./scenarios/utah_flow_scenario_example/utahnetwork.xml" \
"./scenarios/utah_flow_scenario_example/utah_flow_scenario_example_1_000_000/utahplans_1_000_000.xml" \
"--num_agents" "1_000" \