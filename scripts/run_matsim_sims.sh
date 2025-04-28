cd ..

# Set the max RAM to use here
export MAVEN_OPTS="-Xmx61G"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="../cluster_scenarios/utah_flow_scenario_example_10c/utahconfig.xml"
