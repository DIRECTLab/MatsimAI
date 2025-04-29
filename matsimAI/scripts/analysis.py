#!/usr/bin/env python3

import torch
from pathlib import Path
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import itertools
import seaborn as sns
from tbparse import SummaryReader

from matsimAI.flowsim_dataset import FlowSimDataset

sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_tensor_flows(dataset, predicted_flows, target_flows, link_idx, title, save_path):
    link_id = dataset.edge_mapping.inverse[link_idx]
    pred_link_flows = predicted_flows[link_idx].detach().cpu().numpy()
    target_link_flows = target_flows[link_idx].detach().cpu().numpy()

    hours = np.arange(predicted_flows.shape[1])
    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(hours, pred_link_flows, width=bar_width, color="blue", label="Predicted flows")
    plt.bar(hours + bar_width, target_link_flows, width=bar_width, color="red", label="Target flows")
    plt.xlabel("Hour")
    plt.ylabel("Flows")
    plt.title(title + "\nLink Id: " + str(link_id))
    plt.xticks(hours + bar_width / 2, np.arange(1, predicted_flows.shape[1] + 1))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_link_flows(dataset, last_iteration_path, last_iter, link_idx, title, save_path):
    link = dataset.edge_mapping.inverse[link_idx]
    df = pd.read_csv(Path(last_iteration_path, f"{last_iter}.countscompare.txt"), sep="\t")
    df_link = df[df["Link Id"] == int(link)]

    unique_hours = pd.unique(df_link["Hour"])
    bar_width = 0.4
    hours = np.arange(len(unique_hours))

    plt.figure(figsize=(12, 6))
    plt.bar(hours, df_link["MATSIM volumes"], width=bar_width, color="blue", label="Optimized MATSIM volumes")
    plt.bar(hours + bar_width, df_link["Count volumes"], width=bar_width, color="red", label="UTA Count volumes")
    plt.xlabel("Hour")
    plt.ylabel("Volumes")
    plt.title(title + "\nLink Id: " + str(link))
    plt.xticks(hours + bar_width / 2, unique_hours)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tensorboard_scalars(logs_path, save_dir):
    df = SummaryReader(logs_path).scalars

    mad_loss = df[df['tag'] == 'Logs/mad']
    plt.figure()
    plt.plot(mad_loss['step'], mad_loss['value'])
    plt.title("Mean Average Absolute Difference")
    plt.xlabel("Step")
    plt.ylabel("MAAD")
    plt.ylim(500, 1200)
    plt.tight_layout()
    plt.savefig(save_dir / "mad_loss.png")
    plt.close()

    mse_loss = df[df['tag'] == 'Loss/mse']
    plt.figure()
    plt.plot(mse_loss['step'], mse_loss['value'])
    plt.title("Mean Squared Error")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.ylim(1e6, 2.5e6)
    plt.tight_layout()
    plt.savefig(save_dir / "mse_loss.png")
    plt.close()


def plot_clusters(dataset, clusters, save_path):
    edge_idx = dataset.target_graph.edge_index.t().numpy()
    pos = dataset.target_graph.pos.numpy()

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_idx)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_cycle = itertools.cycle(colors)

    node_color_map = {}
    for cluster_id, nodes in clusters.items():
        color = next(color_cycle)
        for node in nodes:
            node_color_map[node] = color

    node_colors = [node_color_map.get(node, 'gray') for node in nx_graph.nodes()]

    plt.figure(figsize=(10, 12))
    nx.draw_networkx(nx_graph, pos, with_labels=False, node_color=node_colors, node_size=100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_clusters(cluster_path, dataset):
    clusters = {}
    with open(cluster_path, "r") as f:
        for line in f.readlines():
            key, vals = line.strip().split(":")
            vals = vals.split(",")[:-1]
            vals = [dataset.node_mapping[v] for v in vals]
            clusters[key] = vals
    return clusters


def main(args):
    results_path = Path(args.results_path)
    matsim_output = Path(args.matsim_output)
    save_dir = Path(args.results_path, "analysis")
    save_dir.mkdir(exist_ok=True)

    last_iter = sorted([int(s.split(".")[1]) for s in os.listdir(matsim_output / "ITERS")])[-1]
    last_iteration_path = Path(matsim_output, "ITERS", f"it.{last_iter}")

    dataset = FlowSimDataset(results_path.parent, args.network_path, args.counts_path, 10)
    target_graph = dataset.target_graph
    sensor_idxs = dataset.sensor_idxs

    flows = torch.load(results_path / "best_flows.pt")
    link_flows = flows["LinkFlows"].to("cpu")
    target_flows = target_graph.edge_attr

    diff = torch.sum(torch.abs(target_flows[sensor_idxs, :] - link_flows[sensor_idxs, :]), dim=1)
    min_sensor_idx = sensor_idxs[torch.argmin(diff)]
    max_sensor_idx = sensor_idxs[torch.argmax(diff)]

    # Plot tensor flow comparisons
    plot_tensor_flows(dataset, link_flows, target_flows, min_sensor_idx, "Minimum Error Link from Gradient Output",
                      save_dir / "min_error_link_gradient.png")
    plot_tensor_flows(dataset, link_flows, target_flows, max_sensor_idx, "Maximum Error Link from Gradient Output",
                      save_dir / "max_error_link_gradient.png")

    # Plot link flows from simulation
    plot_link_flows(dataset, last_iteration_path, last_iter, min_sensor_idx,
                    "Minimum Error Link from Simulation", save_dir / "min_error_link_simulation.png")
    plot_link_flows(dataset, last_iteration_path, last_iter, max_sensor_idx,
                    "Maximum Error Link from Simulation", save_dir / "max_error_link_simulation.png")

    # Plot Tensorboard scalar metrics
    plot_tensorboard_scalars(results_path / "logs", save_dir)

    # Plot clusters
    clusters = load_clusters(results_path / "clusters.txt", dataset)
    plot_clusters(dataset, clusters, save_dir / "network_clusters.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FlowSim results.")
    parser.add_argument("results_path", type=str, help="Path to the results directory.")
    parser.add_argument("network_path", type=str, help="Path to the network.xml file.")
    parser.add_argument("counts_path", type=str, help="Path to the counts.xml file.")
    parser.add_argument("matsim_output", type=str, help="Path to the MATSim output directory.")
    args = parser.parse_args()

    main(args)
