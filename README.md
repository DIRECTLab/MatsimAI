# MATSIMAI

## Project Setup

Make sure you have conda and maven installed. Then to setup the project, change the cuda version to your
version (you can check by running nvidia-smi) and then run the following script, if you don't have a gpu then
don't worry about setting the cuda version.

```
cd matsimAI
./setup_project.sh
```

This compiles the java code with maven, creates a new python environment in conda, and compiles the cython code to activate 
the python environment run

```
conda activate matsimAIenv
```

## Running the Project

The `/scripts` directory contains several example scripts for running the code in this repository:

- `create_populations.sh`:  
  Generates simple populations using a bimodal distribution. Used for generating benchmark results in the associated paper.

- `osm2matsim.sh`:  
  Converts a cleaned OSM network into a MATSim network and parses the `counts.csv` file into a `counts.xml` MATSim format.

- `run_gradient_flow_examples.sh`:  
  Clusters the graph and runs the ADAM optimizer using `network.xml` and `counts.xml`. Generates the traffic assignment matrix if it does not already exist.

- `run_matsim_sims.sh`:  
  Uses `best_plans.xml` from the optimizer to run a MATSim scenario and test the origin-destination plans.

- `run_optimizer_and_matsim.sh`:  
  Runs the optimizer and MATSim simulator sequentially. Suitable for overnight runs. Requires updating the MATSim `config.xml` with correct paths.

- `run_analysis.sh`:  
  Parses simulation and optimization results to produce figures and interactive HTML files for analysis.



## Getting a Real Network in MATSim

### Step 1: Download the Network

1. Visit [JOSM's website](https://josm.openstreetmap.de/) and download the `josm-tested.jar` file.
2. Run JOSM with the following command:

   ```bash
   java -jar josm-tested.jar
   ```

3. The JOSM interface should appear:
   
   ![JOSM home](./figs/josm_home.png)

4. Enable expert mode by clicking on **View** and checking the **Expert mode** box.

5. Navigate to **File → Download Data**. In the download window, switch to the **Download from Overpass API** tab and enter a query.

### Example Overpass API Query

To download a bounding box of road data for the state of Utah, use:

```plaintext
[out:xml];
(
  way["highway"~"motorway|trunk"](39.647,-112.543,41.894,-111.148); //(min latitude, min longitude, max latitude, max longitude)
);
out body;
>;
out skel qt;
```

You can customize this query to include additional roadway types beyond `motorway` and `trunk`. The [Overpass API documentation](https://wiki.openstreetmap.org/wiki/Overpass_API) has more information.

#### Common Road Types:

- **motorway**: Highways or freeways with controlled access.
- **trunk**: Major roads that aren't motorways.
- **primary**, **secondary**, **tertiary**: Roads of varying levels of importance.
- **residential**: Streets within residential areas.
- **living_street**: Streets primarily for pedestrians with limited vehicle access.
- **service**: Roads for accessing buildings, parking lots, etc.
- **footway**, **cycleway**, **path**: Paths for pedestrians and cyclists.
- **track**: Roads mainly used for agricultural or forestry purposes.
- **unclassified**: Roads without a specific classification.

### Step 2: Edit the Network

In the JOSM editor, you can make sure that roads are connected properly. To display the map background, go to **Imagery → OpenStreetMap Carto (Standard)**.

![JOSM editor](figs/josm_editor.png)

Once you're satisfied with the network, save it by going to **File → Save As** and choosing `.osm` as the file format.

## Cleaning the .osm File

Use the script provided in `matsimAI/scripts/` named `clean_osm_data.py` to clean the `.osm` file. An example of how to run the script:

```bash
python -m matsimAI.scripts.clean_osm_data /path/to/osm/data /path/to/output/osm
```

The cleaned `.osm` file can now be converted to a MATSim-compatible `.xml` file.

## Converting .osm to MATSim-Compatible .xml

We've modified the [osm2matsim converter](https://github.com/gustavocovas/osm2matsim), to be able to add sensors while parsing
the network. Its located at `src/main/java/org/matsim/osm2matsim/GetNetworkAndSensors.java`. The script provided at `/scripts/osm2matsim.sh` provides an example on its usage.

# matsim-example-project - from original repository

A small example of how to use MATSim as a library.

By default, this project uses the latest (pre-)release. In order to use a different version, edit `pom.xml`.

A recommended directory structure is as follows:
* `src` for sources
* `original-input-data` for original input data (typically not in MATSim format)
* `scenarios` for MATSim scenarios, i.e. MATSim input and output data.  A good way is the following:
  * One subdirectory for each scenario, e.g. `scenarios/mySpecialScenario01`.
  * This minimally contains a config file, a network file, and a population file.
  * Output goes one level down, e.g. `scenarios/mySpecialScenario01/output-from-a-good-run/...`.
  
  
### Import into eclipse

1. download a modern version of eclipse. This should have maven and git included by default.
1. `file->import->git->projects from git->clone URI` and clone as specified above.  _It will go through a 
sequence of windows; it is important that you import as 'general project'._
1. `file->import->maven->existing maven projects`

Sometimes, step 3 does not work, in particular after previously failed attempts.  Sometimes, it is possible to
right-click to `configure->convert to maven project`.  If that fails, the best thing seems to remove all 
pieces of the failed attempt in the directory and start over.

### Import into IntelliJ

`File -> New -> Project from Version Control` paste the repository url and hit 'clone'. IntelliJ usually figures out
that the project is a maven project. If not: `Right click on pom.xml -> import as maven project`.

### Java Version

The project uses Java 11. Usually a suitable SDK is packaged within IntelliJ or Eclipse. Otherwise, one must install a 
suitable sdk manually, which is available [here](https://openjdk.java.net/)

### Building and Running it locally

You can build an executable jar-file by executing the following command:

```sh
./mvnw clean package
```

or on Windows:

```sh
mvnw.cmd clean package
```

This will download all necessary dependencies (it might take a while the first time it is run) and create a file `matsim-example-project-0.0.1-SNAPSHOT.jar` in the top directory. This jar-file can either be double-clicked to start the MATSim GUI, or executed with Java on the command line:

```sh
java -jar matsim-example-project-0.0.1-SNAPSHOT.jar
```



### Licenses
(The following paragraphs need to be adjusted according to the specifications of your project.)

The **MATSim program code** in this repository is distributed under the terms of the [GNU General Public License as published by the Free Software Foundation (version 2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html). The MATSim program code are files that reside in the `src` directory hierarchy and typically end with `*.java`.

The **MATSim input files, output files, analysis data and visualizations** are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br /> MATSim input files are those that are used as input to run MATSim. They often, but not always, have a header pointing to matsim.org. They typically reside in the `scenarios` directory hierarchy. MATSim output files, analysis data, and visualizations are files generated by MATSim runs, or by postprocessing.  They typically reside in a directory hierarchy starting with `output`.

**Other data files**, in particular in `original-input-data`, have their own individual licenses that need to be individually clarified with the copyright holders.


