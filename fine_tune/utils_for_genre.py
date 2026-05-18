import numpy as np
import pandas as pd
from config import *
from data_utils import *
category_range = [99, 199, 299, 399, 499]

class RandEdgeSampler:
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index], size

class StratifiedEdgeSampler:
    def __init__(self, src_list, dst_list, num_classes=5, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.min_dst = min(dst_list)
        self.max_dst = max(dst_list)
        self.num_classes = num_classes
        self.class_size = (self.max_dst - self.min_dst + 1) // num_classes
        self.class_nodes = []
        for i in range(num_classes):
            start = self.min_dst + i * self.class_size
            end = self.min_dst + (i + 1) * self.class_size if i < num_classes - 1 else self.max_dst + 1
            class_nodes = [n for n in self.dst_list if start <= n < end]
            self.class_nodes.append(np.array(class_nodes) if class_nodes else np.array([]))
        self.non_empty_classes = [i for i in range(num_classes) if len(self.class_nodes[i]) > 0]
        if not self.non_empty_classes:
            raise ValueError("No valid destination nodes found in any class")
        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def sample(self, size):
        if not self.non_empty_classes:
            raise ValueError("No valid classes to sample from")
        total_available_nodes = sum(len(self.class_nodes[idx]) for idx in self.non_empty_classes)
        if total_available_nodes == 0:
            raise ValueError(f"No nodes available for sampling")
        unique_dst_nodes = set()
        for class_idx in self.non_empty_classes:
            for node in self.class_nodes[class_idx]:
                unique_dst_nodes.add(node)
        total_unique_nodes = len(unique_dst_nodes)
        if total_unique_nodes < size:
            print(f"Warning: Requested {size} unique samples, but only {total_unique_nodes} unique nodes available.")
            size = total_unique_nodes
        samples_per_class = size // len(self.non_empty_classes)
        remainder = size % len(self.non_empty_classes)
        src_indices = []
        dst_nodes = []
        sampled_dst_set = set()
        class_samples = {}
        for i, class_idx in enumerate(self.non_empty_classes):
            class_sample_size = samples_per_class + (1 if i < remainder else 0)
            available_nodes = len(self.class_nodes[class_idx])
            if available_nodes == 0:
                class_samples[class_idx] = 0
                continue
            actual_sample_size = min(class_sample_size, available_nodes)
            available_nodes_list = [node for node in self.class_nodes[class_idx] if node not in sampled_dst_set]
            if not available_nodes_list:
                class_samples[class_idx] = 0
                continue
            actual_sample_size = min(actual_sample_size, len(available_nodes_list))
            class_samples[class_idx] = actual_sample_size
            if self.seed is None:
                src_idx = np.random.randint(0, len(self.src_list), actual_sample_size)
                sampled_dst = np.random.choice(available_nodes_list, actual_sample_size, replace=False)
            else:
                src_idx = self.random_state.randint(0, len(self.src_list), actual_sample_size)
                sampled_dst = self.random_state.choice(available_nodes_list, actual_sample_size, replace=False)
            for node in sampled_dst:
                sampled_dst_set.add(node)
            src_indices.append(src_idx)
            dst_nodes.extend(sampled_dst)
        remaining_samples = size - len(dst_nodes)
        if remaining_samples > 0:
            remaining_nodes = [node for class_idx in self.non_empty_classes for node in self.class_nodes[class_idx] if node not in sampled_dst_set]
            if remaining_nodes:
                extra_samples = min(remaining_samples, len(remaining_nodes))
                if self.seed is None:
                    extra_src_idx = np.random.randint(0, len(self.src_list), extra_samples)
                    extra_dst = np.random.choice(remaining_nodes, extra_samples, replace=False)
                else:
                    extra_src_idx = self.random_state.randint(0, len(self.src_list), extra_samples)
                    extra_dst = self.random_state.choice(remaining_nodes, extra_samples, replace=False)
                src_indices.append(extra_src_idx)
                dst_nodes.extend(extra_dst)
        src_indices = np.concatenate(src_indices) if src_indices else np.array([])
        dst_nodes = np.array(dst_nodes)
        actual_size = len(src_indices)
        if len(src_indices) > size:
            src_indices = src_indices[:size]
            dst_nodes = dst_nodes[:size]
            actual_size = size
        if actual_size < size:
            total_unique_nodes = len(unique_dst_nodes)
            if total_unique_nodes >= size:
                error_msg = f"ERROR: Failed to sample {size} nodes despite having {total_unique_nodes} unique nodes available!\n"
                error_msg += f"Only sampled {actual_size} nodes. This indicates a bug in the sampling algorithm."
                raise RuntimeError(error_msg)
            else:
                print(f"Warning: Could only sample {actual_size}/{size} unique nodes (only {total_unique_nodes} unique nodes available).")
        return self.src_list[src_indices], dst_nodes, actual_size

class TemporalStratifiedEdgeSampler:
    def __init__(self, src_list, dst_list, timestamps, num_dst_classes=5, num_time_bins=5, seed=None):
        self.seed = seed
        assert len(src_list) == len(dst_list) == len(timestamps), "Input arrays must have the same length"
        self.src_list = np.unique(src_list)
        self.min_dst = min(dst_list)
        self.max_dst = max(dst_list)
        self.num_dst_classes = num_dst_classes
        self.dst_class_size = (self.max_dst - self.min_dst + 1) // num_dst_classes
        self.num_time_bins = num_time_bins
        self.min_time = min(timestamps) if len(timestamps) > 0 else 0
        self.max_time = max(timestamps) if len(timestamps) > 0 else 0
        self.time_bin_size = (self.max_time - self.min_time) / num_time_bins if self.max_time > self.min_time else 1
        self.stratified_edges = {}
        for i in range(len(src_list)):
            src, dst, timestamp = src_list[i], dst_list[i], timestamps[i]
            dst_class, time_bin = self._get_dst_class(dst), self._get_time_bin(timestamp)
            key = (dst_class, time_bin)
            if key not in self.stratified_edges:
                self.stratified_edges[key] = []
            self.stratified_edges[key].append((src, dst, timestamp))
        self.valid_bins = [k for k, v in self.stratified_edges.items() if len(v) > 0 and k[0] != -1 and k[1] != -1]
        if not self.valid_bins and len(src_list) > 0:
            key = (0, 0)
            self.stratified_edges[key] = list(zip(src_list, dst_list, timestamps))
            self.valid_bins = [key]
        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def _get_dst_class(self, dst_node):
        if not (self.min_dst <= dst_node <= self.max_dst):
            return -1
        return min(self.num_dst_classes - 1, (dst_node - self.min_dst) // self.dst_class_size)

    def _get_time_bin(self, timestamp):
        if self.max_time == self.min_time:
            return 0
        if not (self.min_time <= timestamp <= self.max_time):
            return -1
        return min(self.num_time_bins - 1, int((timestamp - self.min_time) / self.time_bin_size))

    def sample(self, size):
        if not self.valid_bins:
            raise ValueError("No valid bins to sample from")
        bin_sizes = {key: len(self.stratified_edges[key]) for key in self.valid_bins}
        total_edges = sum(bin_sizes.values())
        if total_edges == 0:
            raise ValueError("No edges available for sampling")
        unique_edges = {edge for key in self.valid_bins for edge in self.stratified_edges[key]}
        total_unique_edges = len(unique_edges)
        if total_unique_edges < size:
            print(f"Warning: Requested {size} unique samples, but only {total_unique_edges} unique edges available.")
            size = total_unique_edges
        samples_per_bin = size // len(self.valid_bins)
        remainder = size % len(self.valid_bins)
        src_nodes, dst_nodes, timestamps = [], [], []
        sampled_edges_set = set()
        for i, key in enumerate(self.valid_bins):
            bin_sample_size = samples_per_bin + (1 if i < remainder else 0)
            available_edges_indices = [idx for idx, edge in enumerate(self.stratified_edges[key]) if edge not in sampled_edges_set]
            if not available_edges_indices:
                continue
            actual_sample_size = min(bin_sample_size, len(available_edges_indices))
            if self.seed is None:
                indices = np.random.choice(available_edges_indices, actual_sample_size, replace=False)
            else:
                indices = self.random_state.choice(available_edges_indices, actual_sample_size, replace=False)
            for idx in indices:
                u, i_node, t = self.stratified_edges[key][idx]
                sampled_edges_set.add((u, i_node, t))
                src_nodes.append(u)
                dst_nodes.append(i_node)
                timestamps.append(t)
        remaining_samples = size - len(src_nodes)
        if remaining_samples > 0:
            remaining_edges = [(key, idx) for key in self.valid_bins for idx, edge in enumerate(self.stratified_edges[key]) if edge not in sampled_edges_set]
            if remaining_edges:
                extra_samples = min(remaining_samples, len(remaining_edges))
                if self.seed is None:
                    selected_indices = np.random.choice(len(remaining_edges), extra_samples, replace=False)
                else:
                    selected_indices = self.random_state.choice(len(remaining_edges), extra_samples, replace=False)
                for idx in selected_indices:
                    key, edge_idx = remaining_edges[idx]
                    u, i_node, t = self.stratified_edges[key][edge_idx]
                    sampled_edges_set.add((u, i_node, t))
                    src_nodes.append(u)
                    dst_nodes.append(i_node)
                    timestamps.append(t)
        actual_size = len(src_nodes)
        if actual_size > size:
            src_nodes, dst_nodes, timestamps = src_nodes[:size], dst_nodes[:size], timestamps[:size]
            actual_size = size
        if actual_size < size:
            if total_unique_edges >= size:
                error_msg = f"ERROR: Failed to sample {size} edges despite having {total_unique_edges} unique edges available!\n"
                error_msg += f"Only sampled {actual_size} edges. This indicates a bug in the sampling algorithm."
                raise RuntimeError(error_msg)
            else:
                print(f"Warning: Could only sample {actual_size}/{size} unique edges (only {total_unique_edges} unique edges available).")
        return np.array(src_nodes), np.array(dst_nodes), np.array(timestamps), actual_size

if __name__ == "__main__":
    args = get_args()
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_d_data(args, args.dataset, args.different_new_nodes)
    src_list, dst_list, timestamps = full_data.sources, full_data.destinations, full_data.timestamps
    print("\n===== Basic Dataset Information =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total number of destination nodes: {len(np.unique(dst_list))}")
    print(f"Minimum destination node: {min(dst_list)}")
    print(f"Maximum destination node: {max(dst_list)}")
    print(f"Total number of edges: {len(dst_list)}")
    num_dst_classes, num_time_bins = 5, 5
    print("\n===== Testing StratifiedEdgeSampler =====")
    stratified_sampler = StratifiedEdgeSampler(src_list, dst_list, num_classes=num_dst_classes, seed=42)
    print(f"Class size: {stratified_sampler.class_size}")
    print(f"Non-empty classes: {stratified_sampler.non_empty_classes}")
    for i, nodes in enumerate(stratified_sampler.class_nodes):
        print(f"Class {i}: {len(nodes)} nodes")
    sample_size = 100
    print(f"\nSampling {sample_size} samples...")
    sampled_src, sampled_dst, actual_size = stratified_sampler.sample(sample_size)
    print(f"Actual number of samples: {actual_size}")
    class_counts = [0] * num_dst_classes
    for dst in sampled_dst:
        class_idx = (dst - stratified_sampler.min_dst) // stratified_sampler.class_size
        class_idx = min(class_idx, num_dst_classes - 1)
        class_counts[class_idx] += 1
    print("Number of samples in each class from the sampling result:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count/actual_size*100:.2f}%)")
    print("\n===== Testing TemporalStratifiedEdgeSampler =====")
    temporal_sampler = TemporalStratifiedEdgeSampler(src_list, dst_list, timestamps, num_dst_classes=num_dst_classes, num_time_bins=num_time_bins, seed=42)
    print(f"Time bin size: {temporal_sampler.time_bin_size}")
    print(f"Number of valid bins: {len(temporal_sampler.valid_bins)}")
    print("\nNumber of edges in valid bins:")
    for key in temporal_sampler.valid_bins:
        dst_class, time_bin = key
        print(f"Destination class {dst_class}, Time bin {time_bin}: {len(temporal_sampler.stratified_edges[key])} edges")
    sample_size = 100
    print(f"\nSampling {sample_size} samples...")
    sampled_src, sampled_dst, sampled_times, actual_size = temporal_sampler.sample(sample_size)
    print(f"Actual number of samples: {actual_size}")
    class_time_counts = np.zeros((num_dst_classes, num_time_bins), dtype=int)
    for i in range(len(sampled_dst)):
        dst, t = sampled_dst[i], sampled_times[i]
        dst_class, time_bin = temporal_sampler._get_dst_class(dst), temporal_sampler._get_time_bin(t)
        if dst_class != -1 and time_bin != -1:
            class_time_counts[dst_class, time_bin] += 1
    print("\nNumber of edges in class-time bins from the sampling result:")
    print("Class\\Time Bin", end="")
    for i in range(num_time_bins):
        print(f"\tBin {i}", end="")
    print("\tTotal")
    for i in range(num_dst_classes):
        print(f"Class {i}", end="")
        row_sum = 0
        for j in range(num_time_bins):
            count = class_time_counts[i, j]
            row_sum += count
            print(f"\t{count}", end="")
        print(f"\t{row_sum}")
    print("Total", end="")
    for j in range(num_time_bins):
        col_sum = sum(class_time_counts[:, j])
        print(f"\t{col_sum}", end="")
    print(f"\t{sum(sum(class_time_counts))}")
