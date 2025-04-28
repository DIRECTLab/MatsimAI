cd ..

mvn exec:java -Dexec.mainClass="org.matsim.osm2matsim.GetNetworkAndSensors" \
 -Dexec.args="matsimAI/udot-sensors/utah_network_cleaned.osm \
 scenarios/utah_flow_scenario_example/utahnetwork.xml \
 matsimAI/udot-sensors/sensor_data.csv \
 scenarios/utah_flow_scenario_example/utahcounts.xml \
 100"
