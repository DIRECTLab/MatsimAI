cd ..

# Set the max RAM to use here
export MAVEN_OPTS="-Xmx61G"

mvn exec:java -Dexec.mainClass="org.matsim.run.RunMatsim" -Dexec.args="./scenarios/cluster_scenarios/utah_100c/utahconfig.xml"
