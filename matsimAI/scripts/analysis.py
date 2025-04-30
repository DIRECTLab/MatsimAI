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
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from bokeh.plotting import figure, output_file, save
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, HoverTool, CustomJS, Select
from bokeh.layouts import row, column
from bokeh.io import output_notebook
from bokeh.palettes import RdYlGn11
from bokeh.transform import linear_cmap

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
    plt.bar(hours + bar_width, target_link_flows, width=bar_width, color="red", label="Count Volumes")
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
    plt.bar(hours, df_link["MATSIM volumes"], width=bar_width, color="blue", label="MATSIM volumes")
    plt.bar(hours + bar_width, df_link["Count volumes"], width=bar_width, color="red", label="Count volumes")
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

def create_html_plot(dataset, link_flows, hour_count, title, output_html_path="sensor_edges_graph.html"):
    sensor_idxs = dataset.sensor_idxs
    target_flows = dataset.target_graph.edge_attr
    pos = {i: (dataset.target_graph.pos[i][0].item(), dataset.target_graph.pos[i][1].item()) for i in range(len(dataset.target_graph.pos))}

    G_sensor = nx.Graph()

    for i in range(dataset.target_graph.num_edges):
        u, v = dataset.target_graph.edge_index[0][i].item(), dataset.target_graph.edge_index[1][i].item()
        
        if i in sensor_idxs:
            edge_attrs = {"Absolute Difference": [], "Predicted Flow": [], "Target Flow": []}

            for hour_idx in range(hour_count):
                if isinstance(link_flows, torch.Tensor):
                    target_flow = target_flows[i][hour_idx].item()
                    pred_flow = link_flows[i][hour_idx].item()
                elif isinstance(link_flows, pd.DataFrame):
                    link_id = int(dataset.edge_mapping.inv[i])
                    df_row = link_flows[(link_flows['Link Id'] == link_id) & (link_flows['Hour'] == (hour_idx+1))].iloc[0]
                    target_flow = df_row['Count volumes']
                    pred_flow = df_row['MATSIM volumes']
                else:
                    raise ValueError("link_flows must be either a torch.Tensor or a pd.DataFrame")

                abs_diff = abs(target_flow - pred_flow)

                edge_attrs["Absolute Difference"].append(abs_diff)
                edge_attrs["Predicted Flow"].append(pred_flow)
                edge_attrs["Target Flow"].append(target_flow)

            G_sensor.add_edge(u, v, **edge_attrs)
        else:
            G_sensor.add_edge(u, v)

    # Compute max absolute difference for each hour
    max_abs_diff_per_hour = [0] * hour_count
    for hour_idx in range(hour_count):
        max_abs_diff_per_hour[hour_idx] = max(
            [max([data["Absolute Difference"][hour_idx] for u, v, data in G_sensor.edges(data=True) if "Absolute Difference" in data], default=0)],
            default=1e-6  # avoid zero division
        )

    # Build plot data
    sensor_edges = []
    normal_edges = []
    weight_all_hours, pred_all_hours, target_all_hours = [], [], []

    for (u, v, data) in tqdm(G_sensor.edges(data=True), desc="Processing Sensor Edges"):
        if "Absolute Difference" in data:
            sensor_edges.append((u, v, data))
        else:
            normal_edges.append((u, v))

    # Sensor edges (colored)
    s_x0s, s_y0s, s_x1s, s_y1s = [], [], [], []
    mid_x, mid_y = [], []
    for (u, v, data) in sensor_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        s_x0s.append(x0)
        s_y0s.append(y0)
        s_x1s.append(x1)
        s_y1s.append(y1)
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        weight_all_hours.append(data["Absolute Difference"])
        pred_all_hours.append(data["Predicted Flow"])
        target_all_hours.append(data["Target Flow"])

    sensors_edge_source = ColumnDataSource(data=dict(
        x0=s_x0s, y0=s_y0s, x1=s_x1s, y1=s_y1s,
        mid_x=mid_x, mid_y=mid_y,
        weight=[w[0] for w in weight_all_hours],
        predicted=[p[0] for p in pred_all_hours],
        target=[t[0] for t in target_all_hours],
        weight_all_hours=weight_all_hours,
        predicted_all_hours=pred_all_hours,
        target_all_hours=target_all_hours,
    ))

    # Normal edges (black)
    n_x0s, n_y0s, n_x1s, n_y1s = [], [], [], []
    for (u, v) in normal_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        n_x0s.append(x0)
        n_y0s.append(y0)
        n_x1s.append(x1)
        n_y1s.append(y1)

    normal_edge_source = ColumnDataSource(data=dict(
        x0=n_x0s, y0=n_y0s, x1=n_x1s, y1=n_y1s,
    ))

    # Create Bokeh figure
    plot = figure(title="Absolute Flow Difference", width=800, height=600, tools="pan,wheel_zoom,box_zoom,reset,save")

    # Loop through each hour to create a color mapper per hour
    for hour_idx in range(hour_count):
        # Color mapper for each hour
        # color_mapper = LinearColorMapper(palette=RdYlGn11, low=0, high=max_abs_diff_per_hour[hour_idx])
        color_mapper = LinearColorMapper(palette=RdYlGn11, low=0, high=max(max_abs_diff_per_hour))

        # Draw sensor edges with color
        plot.segment('x0', 'y0', 'x1', 'y1', source=sensors_edge_source,
                     line_width=10, color={'field': 'weight', 'transform': color_mapper})

    # Draw normal edges in black
    plot.segment('x0', 'y0', 'x1', 'y1', source=normal_edge_source,
                 line_width=1, color="black")

    # Draw nodes
    node_x = [pos[i][0] for i in pos]
    node_y = [pos[i][1] for i in pos]
    plot.scatter(node_x, node_y, size=.01, color="black", alpha=0.1)

    # Hover tool
    tooltips = [("Abs Diff", "@weight"), ("Predicted", "@predicted"), ("Target", "@target")]
    hover = HoverTool(tooltips=tooltips)
    plot.add_tools(hover)

    # Colorbar
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    plot.add_layout(color_bar, 'right')


    # === Histogram Setup ===
    bin_edges = np.linspace(0, max(max_abs_diff_per_hour), 21)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    hist_source = ColumnDataSource(data=dict(
        top=[0]*20,
        bin_center=bin_centers
    ))

    hist_plot = figure(title="Visible Edge Flow Difference Histogram", width=300, height=600, y_range=(0, max(max_abs_diff_per_hour)),
                       toolbar_location=None, tools="")
    hist_plot.hbar(y='bin_center', height=(bin_edges[1] - bin_edges[0]) * 0.8, right='top', left=0, source=hist_source,
                   fill_color=linear_cmap('bin_center', RdYlGn11, low=0, high=max(max_abs_diff_per_hour)), line_color="white")
    hist_plot.xaxis.axis_label = "Count"
    hist_plot.yaxis.axis_label = "Abs Diff"
    hist_plot.ygrid.grid_line_color = None

    # Dropdown for hour selection
    hour_selector = Select(title="Select Hour", value="0", options=[str(i) for i in range(hour_count)])

    # === JS Callback ===
    callback = CustomJS(args=dict(
        source=sensors_edge_source,
        hist_source=hist_source,
        x_range=plot.x_range,
        y_range=plot.y_range,
        hour_selector=hour_selector
    ), code="""
        const hour = parseInt(hour_selector.value);
        const x0 = x_range.start;
        const x1 = x_range.end;
        const y0 = y_range.start;
        const y1 = y_range.end;

        const mids_x = source.data['mid_x'];
        const mids_y = source.data['mid_y'];
        const weights_all = source.data['weight_all_hours'];
        const preds_all = source.data['predicted_all_hours'];
        const targets_all = source.data['target_all_hours'];

        const weights = [];
        for (let i = 0; i < mids_x.length; i++) {
            if (mids_x[i] >= x0 && mids_x[i] <= x1 && mids_y[i] >= y0 && mids_y[i] <= y1) {
                weights.push(weights_all[i][hour]);
            }
            source.data['weight'][i] = weights_all[i][hour];
            source.data['predicted'][i] = preds_all[i][hour];
            source.data['target'][i] = targets_all[i][hour];
        }

        const bin_count = 20;
        const bin_min = 0;
        const bin_max = Math.max(...weights_all.flat());
        const bin_size = (bin_max - bin_min) / bin_count;

        const hist = Array(bin_count).fill(0);
        const bin_centers = [];
        for (let i = 0; i < bin_count; i++) {
            bin_centers.push(bin_min + bin_size * (i + 0.5));
        }

        for (let i = 0; i < weights.length; i++) {
            const w = weights[i];
            const bin = Math.floor((w - bin_min) / bin_size);
            if (bin >= 0 && bin < bin_count) {
                hist[bin]++;
            }
        }

        hist_source.data['top'] = hist;
        hist_source.data['bin_center'] = bin_centers;

        source.change.emit();
        hist_source.change.emit();
    """)

    # Link all events to callback
    plot.x_range.js_on_change("start", callback)
    plot.x_range.js_on_change("end", callback)
    plot.y_range.js_on_change("start", callback)
    plot.y_range.js_on_change("end", callback)
    hour_selector.js_on_change('value', callback)

    layout = row(plot, hist_plot, column(hour_selector))
    save(layout, filename=output_html_path, title=title)

def build_abs_diff_graph(dataset, link_flows, sensor_idxs, target_flows, hour, title, save_path):
    hour_idx = hour
    hour_val = hour + 1
    G_sensor = nx.Graph()
    # Add edges with full attributes
    for i in range(dataset.target_graph.num_edges):
        u, v = dataset.target_graph.edge_index[0][i].item(), dataset.target_graph.edge_index[1][i].item()
        
        if i in sensor_idxs:
            if isinstance(link_flows, torch.Tensor):
                target_flow = target_flows[i][hour_idx].item()
                pred_flow = link_flows[i][hour_idx].item()

                abs_diff = abs(target_flow - pred_flow)

                G_sensor.add_edge(u, v, **{'Absolute Difference':abs_diff, 'Predicted Flow': pred_flow, 'Target Flow': target_flow})
            elif isinstance(link_flows, pd.DataFrame):
                link_id = int(dataset.edge_mapping.inv[i])
                df_row = link_flows[(link_flows['Link Id'] == link_id) & (link_flows['Hour'] == hour_val)].iloc[0]
                target_flow = df_row['Count volumes']
                pred_flow = df_row['MATSIM volumes']
                abs_diff = abs(target_flow - pred_flow)
                G_sensor.add_edge(u, v, **{'Absolute Difference':abs_diff, 'Predicted Flow': pred_flow, 'Target Flow': target_flow})
            else:
                raise ValueError("link_flows must be either a torch.Tensor or a pd.DataFrame")
        else:
            G_sensor.add_edge(u, v)

    # Prepare plotting
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 5, width_ratios=[50, -15, 1.5, 0.0, 10])  # wider graph area

    ax_graph = fig.add_subplot(gs[0])  # Main graph
    ax_hist = fig.add_subplot(gs[4])  # Histogram

    # Get positions for nodes
    pos = {i: (dataset.target_graph.pos[i][0].item(), dataset.target_graph.pos[i][1].item()) for i in range(len(dataset.target_graph.pos))}

    # Separate edges
    edges_with_sensor = [(u, v) for u, v in G_sensor.edges() if 'Absolute Difference' in G_sensor[u][v]]
    edges_without_sensor = [(u, v) for u, v in G_sensor.edges() if 'Absolute Difference' not in G_sensor[u][v]]

    # Extract weights
    weights = [G_sensor[u][v]['Absolute Difference'] for u, v in edges_with_sensor]
    if len(weights) == 0:
        weights = [0]

    # Normalize weights
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.cm.RdYlGn.reversed()
    edge_colors_sensor = [cmap(norm(weight)) for weight in weights]

    # Draw graph
    nx.draw_networkx_edges(G_sensor, pos, edgelist=edges_with_sensor, edge_color=edge_colors_sensor, width=20, ax=ax_graph)
    nx.draw_networkx_edges(G_sensor, pos, edgelist=edges_with_sensor, edge_color='black', width=1, ax=ax_graph)
    nx.draw_networkx_edges(G_sensor, pos, edgelist=edges_without_sensor, edge_color='black', width=1, ax=ax_graph)
    nx.draw_networkx_nodes(G_sensor, pos, node_size=1, node_color='black', ax=ax_graph)

    ax_graph.set_xlabel("X Coordinate")
    ax_graph.set_ylabel("Y Coordinate")
    ax_graph.set_axis_off()
    ax_graph.set_title(title)

    # Histogram
    hist_vals, bins = np.histogram(weights, bins=20)
    bar_centers = (bins[:-1] + bins[1:]) / 2
    bar_heights = hist_vals
    bar_width = (bins[1] - bins[0]) * 0.9

    # Map bin centers to colors
    bin_centers = (bins[:-1] + bins[1:]) / 2  # middle of each bin
    colors = [cmap(norm(center)) for center in bin_centers]

    ax_hist.barh(bar_centers, bar_heights, height=bar_width, align='center', color=colors, edgecolor='black')

    # Add count labels at the top of each bar
    for count, center in zip(bar_heights, bar_centers):
        ax_hist.text(count + 1, center, str(count), va='center', ha='left', fontsize=8)

    ax_hist.set_xlabel('Count')
    ax_hist.set_ylabel('')

    # Adjust layout
    plt.tight_layout()
    plt.title("Absolute Difference \n of Sensor Flows")
    plt.savefig(save_path)
    plt.close()


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

    # flows = torch.load(results_path / "best_flows.pt")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    flows = torch.load(results_path / "best_flows.pt", map_location=device)
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


    gradient_abs_diff_path = Path(save_dir, "gradient_abs_diff")
    gradient_abs_diff_path.mkdir(exist_ok=True)

    simulation_abs_diff_path = Path(save_dir, "simulation_abs_diff")
    simulation_abs_diff_path.mkdir(exist_ok=True)

    link_flows_df = pd.read_csv(Path(last_iteration_path, f"{last_iter}.countscompare.txt"), sep="\t")
    for hour in tqdm(range(24), desc="Creating Absolute Difference Graphs"):
        build_abs_diff_graph(dataset, link_flows, sensor_idxs, target_flows, hour,
                        f"Absolute Difference of Gradient Sensor Flows at Hour {hour+1}",
                        gradient_abs_diff_path / f"abs_diff_graph_hour_{hour+1}.png")

        build_abs_diff_graph(dataset, link_flows_df, sensor_idxs, target_flows, hour,
                        f"Absolute Difference of Simulation Sensor Flows at Hour {hour+1}",
                        simulation_abs_diff_path / f"abs_diff_graph_hour_{hour+1}.png")
    create_html_plot(dataset, link_flows, 24, "Gradient Results", Path(save_dir, "gradient_results.html"))
    create_html_plot(dataset, link_flows_df, 24, "Simulation Results", Path(save_dir, "simulation_results.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FlowSim results.")
    parser.add_argument("results_path", type=str, help="Path to the results directory.")
    parser.add_argument("network_path", type=str, help="Path to the network.xml file.")
    parser.add_argument("counts_path", type=str, help="Path to the counts.xml file.")
    parser.add_argument("matsim_output", type=str, help="Path to the MATSim output directory.")
    args = parser.parse_args()

    main(args)
