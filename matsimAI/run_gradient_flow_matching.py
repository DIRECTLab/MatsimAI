import argparse
from matsimAI.flowsim_dataset import FlowSimDataset
from pathlib import Path
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom

def main(args):
    
    current_time = datetime.datetime.now()
    unique_time_string = current_time.strftime("%m%d%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_path = Path(args.results_path)
    network_name = Path(args.network_path).stem
    save_path = Path(output_path, f"{unique_time_string}_nclusters_{args.num_clusters}_{network_name}")
    tensorboard_path = Path(save_path, "logs")
    writer = SummaryWriter(tensorboard_path)


    os.makedirs(tensorboard_path)
    
    tree = ET.parse(args.config_path)
    root = tree.getroot()
    for module in root.findall("module"):
        if module.attrib.get("name") == "counts":
            for param in module.findall("param"):
                if param.attrib.get("name") == "countsScaleFactor":
                    param.attrib["value"] = str(args.percent_pop)
            else:
                ET.SubElement(module, "param", {"name": "countsScaleFactor", "value": str(args.percent_pop)})

    temp_path = "temp_output.xml"
    tree.write(temp_path, encoding="utf-8", xml_declaration=False)

    with open(temp_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    with open(args.config_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">\n')
        f.write(xml_content)

    os.remove(temp_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset = FlowSimDataset(output_path, args.network_path, args.counts_path, args.num_clusters)
    dataset.save_clusters(Path(save_path, "clusters.txt"))

    with open(Path(save_path, "sensor_ids.txt"), "w") as f:
        for sensor_idx in dataset.sensor_idxs:
            f.write(f"{dataset.edge_mapping.inv[sensor_idx]},")
    
    sensor_idxs = dataset.sensor_idxs
    
    with open(Path(save_path, "args.txt"), "w") as f:
        for key, val in args.__dict__.items():
            f.write(f"{key}:{val}\n")
        f.write(f"num_nodes:{dataset.target_graph.num_nodes}\n")
        f.write(f"num_edges:{dataset.target_graph.num_edges}\n")
        f.write(f"num_sensors:{len(sensor_idxs)}\n")

    Z_2 = args.num_clusters**2
    TAM = torch.from_numpy(dataset.TAM).to(device).to(torch.float32)
    W = torch.nn.Parameter(torch.rand(Z_2, 24).to(device).to(torch.float32))
    parameters = [W]

    TAM = TAM.reshape(-1, Z_2)
    percent_pop_float = float(args.percent_pop) / 100
    TARGET = dataset.target_graph.edge_attr.to(device).to(torch.float32) * percent_pop_float

    optimizer = torch.optim.Adam(parameters, lr=0.001)
    pbar = tqdm(range(args.training_steps))
    target_size = TARGET[sensor_idxs].numel()

    best_loss = torch.inf
    best_model = None

    for step in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            W.data.clamp_(0, torch.inf)
        R = torch.matmul(TAM, W)
        loss = torch.nn.functional.mse_loss(R[sensor_idxs], TARGET[sensor_idxs])
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = W.clone()

        if step % args.log_interval == 0:
            pbar.set_description(f"Loss: {loss.item()}")
            writer.add_scalar("Loss/mse", loss.item(), step)
            writer.add_scalar("Logs/mad", torch.abs(R[sensor_idxs] - (TARGET[sensor_idxs])).sum() / target_size, step)

        if step != 0 and \
        args.save_interval > 0 and \
        step % args.save_interval == 0:
            if best_model is not None:
                torch.save({'OD': best_model, 'LinkFlows': R}, Path(save_path, f"best_flows.pt"))
                dataset.save_plans_from_flow_res(
                    best_model.reshape(args.num_clusters, args.num_clusters, 24),
                    Path(save_path, "best_plans.xml")
                )

    with torch.no_grad():
        W.data.clamp_(0, torch.inf)

    if best_model is not None:
        torch.save({'OD': best_model, 'LinkFlows': R}, Path(save_path, f"best_flows.pt"))
        dataset.save_plans_from_flow_res(
            best_model.reshape(args.num_clusters, args.num_clusters, 24),
            Path(save_path, "best_plans.xml")
        )
    
    if args.best_plans_save_path is not None:
        dataset.save_plans_from_flow_res(
            best_model.reshape(args.num_clusters, args.num_clusters, 24),
            Path(args.best_plans_save_path)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_path", help="path to the output folder for the results of the algorithm")
    parser.add_argument("network_path", help="path to matsim xml network")
    parser.add_argument("counts_path", help="path to matsim xml counts")
    parser.add_argument("config_path", help="path to matsim xml config path")
    parser.add_argument("--percent_pop", type=str, default=100, help="The percentage of the population to match to between 0-100.")
    parser.add_argument("--num_clusters", type=int, required=True, help="number of clusters for the network")
    parser.add_argument("--training_steps", type=int, required=True, help="number of training iterations")
    parser.add_argument("--log_interval", type=int, required=True, help="tensorboard logging interval")
    parser.add_argument("--save_interval", type=int, required=True, help="model save interval")
    parser.add_argument("--best_plans_save_path", type=str, default=None, help="addtional path to save the best plans, helpful for\
                        automating matsim runs")

    args = parser.parse_args()
    main(args)
