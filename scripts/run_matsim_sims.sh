cd ..

# Set the max RAM to use here
export MAVEN_OPTS="-Xmx61G"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/utah_flow_scenario_example/utahconfig.xml"
