#!/usr/bin/env python3
"""Generate publication figures for the Dream Layer v0.1 paper."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

RESULTS_DIR = Path.home() / "Documents" / "AIProjects" / "LAC" / "tcd" / "potential_phd" / "experiments" / "results"
FIG_DIR = Path(__file__).parent

MODES = {
    'random':    RESULTS_DIR / "ablation_random",
    'template':  RESULTS_DIR / "ablation_template",
    'structure': RESULTS_DIR / "ablation_structure",
    'feedback':  RESULTS_DIR / "concept_learner",
}

MODE_LABELS = {
    'random': 'Random',
    'template': 'Template',
    'structure': 'Structure',
    'feedback': 'Feedback (v6)',
}

COLORS = {
    'random':    '#6c757d',   # grey
    'template':  '#fd7e14',   # orange
    'structure': '#0d6efd',   # blue
    'feedback':  '#198754',   # green
}


def load_data():
    """Load all ablation results."""
    data = {}
    for mode, path in MODES.items():
        state_file = path / 'dream_state.json'
        session_file = path / 'session_0001.json'

        if not session_file.exists():
            print(f"  WARNING: {session_file} not found, skipping {mode}")
            continue

        with open(session_file) as f:
            log = json.load(f)

        active = [e for e in log if e.get('mode') == 'active']
        igs = [e.get('info_gain', 0) for e in active if 'info_gain' in e]

        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            n_concepts = len(state.get('concepts', []))
            n_connections = len(state.get('concept_graph_edges', []))
        else:
            # Extract from session log if state file missing
            state = {}
            # Use last entry's concept count if available
            n_concepts = 20  # known from RESEARCH_LOG
            # Known values from ablation results table
            known_connections = {'random': 119, 'template': 70, 'structure': 62, 'feedback': 301}
            n_connections = known_connections.get(mode, 0)
            print(f"  WARNING: {state_file} not found, using known values for {mode}")

        data[mode] = {
            'state': state,
            'log': log,
            'n_concepts': n_concepts,
            'n_connections': n_connections,
            'n_active': len(active),
            'n_passive': len(log) - len(active),
            'avg_ig': np.mean(igs) if igs else 0,
            'max_ig': max(igs) if igs else 0,
            'igs': igs,
            'conn_per_concept': n_connections / max(n_concepts, 1),
        }
    return data


def figure1_connections_per_concept(data):
    """Bar chart: connections per concept across 4 modes."""
    fig, ax = plt.subplots(figsize=(6, 4))

    modes = ['random', 'template', 'structure', 'feedback']
    values = [data[m]['conn_per_concept'] for m in modes]
    colors = [COLORS[m] for m in modes]
    labels = [MODE_LABELS[m] for m in modes]

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Connections per Concept')
    ax.set_title('Knowledge Graph Density by Question Strategy')
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation for the key finding
    ax.annotate('4.9x', xy=(3, values[3]), xytext=(2.2, values[3] * 0.7),
                arrowprops=dict(arrowstyle='->', color='#198754', lw=1.5),
                fontsize=14, fontweight='bold', color='#198754',
                ha='center')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_connections.pdf')
    fig.savefig(FIG_DIR / 'fig1_connections.png')
    plt.close(fig)
    print("  Figure 1: fig1_connections.pdf")


def figure2_cumulative_ig(data):
    """Line chart: cumulative information gain over 200 steps."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode in ['random', 'template', 'structure', 'feedback']:
        log = data[mode]['log']
        # Build cumulative IG over ALL steps (0 for passive)
        cum_ig = []
        total = 0
        for entry in log:
            if entry.get('mode') == 'active' and 'info_gain' in entry:
                total += entry['info_gain']
            cum_ig.append(total)

        steps = range(1, len(cum_ig) + 1)
        ax.plot(steps, cum_ig, label=MODE_LABELS[mode], color=COLORS[mode],
                linewidth=2 if mode == 'feedback' else 1.2,
                alpha=1.0 if mode == 'feedback' else 0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Information Gain')
    ax.set_title('Learning Rate Across Question Strategies')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_cumulative_ig.pdf')
    fig.savefig(FIG_DIR / 'fig2_cumulative_ig.png')
    plt.close(fig)
    print("  Figure 2: fig2_cumulative_ig.pdf")


def figure3_strategy_breakdown(data):
    """Grouped bar chart: strategy-level IG comparison (structure vs feedback)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    strategies = ['connection', 'shared_feat', 'feature_gap', 'unvisited', 'combine']
    strategy_labels = ['Connection', 'Shared\nFeature', 'Feature\nGap', 'Unvisited', 'Combine']

    # Get strategy-level data
    struct_igs = {}
    feed_igs = {}

    for mode_name, mode_key in [('structure', 'structure'), ('feedback', 'feedback')]:
        state = data[mode_key]['state']
        scores = state.get('strategy_scores', {})
        for s in strategies:
            vals = scores.get(s, [])
            avg = np.mean(vals) if vals else 0
            if mode_name == 'structure':
                struct_igs[s] = avg
            else:
                feed_igs[s] = avg

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, [struct_igs.get(s, 0) for s in strategies],
                   width, label='Structure (no feedback)', color=COLORS['structure'], alpha=0.8)
    bars2 = ax.bar(x + width/2, [feed_igs.get(s, 0) for s in strategies],
                   width, label='Feedback (v6)', color=COLORS['feedback'], alpha=0.8)

    # Add improvement percentages
    for i, s in enumerate(strategies):
        sv = struct_igs.get(s, 0)
        fv = feed_igs.get(s, 0)
        if sv > 0:
            pct = ((fv - sv) / sv) * 100
            ax.text(x[i] + width/2, fv + 0.3, f'+{pct:.0f}%',
                    ha='center', va='bottom', fontsize=8, color='#198754', fontweight='bold')

    ax.set_ylabel('Average Information Gain')
    ax.set_title('Strategy Effectiveness: Structure vs Feedback')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_strategy_breakdown.pdf')
    fig.savefig(FIG_DIR / 'fig3_strategy_breakdown.png')
    plt.close(fig)
    print("  Figure 3: fig3_strategy_breakdown.pdf")


def figure4_ig_acceleration():
    """Bar chart: average IG per 400-step window across 2000 steps."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Data from RESEARCH_LOG.md (2000-step extended run)
    windows = ['0-400', '400-800', '800-1200', '1200-1600', '1600-2000']
    avg_igs = [15.1, 38.3, 61.7, 79.8, 100.3]

    bars = ax.bar(windows, avg_igs, color='#198754', edgecolor='white',
                  linewidth=0.5, width=0.6, alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars, avg_igs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Step Window')
    ax.set_ylabel('Average Information Gain')
    ax.set_title('Information Gain Acceleration (2000-Step Run)')
    ax.set_ylim(0, max(avg_igs) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add trend annotation
    ax.annotate('6.6x acceleration', xy=(4, avg_igs[4]), xytext=(2.5, avg_igs[4] * 0.85),
                arrowprops=dict(arrowstyle='->', color='#198754', lw=1.5),
                fontsize=12, fontweight='bold', color='#198754', ha='center')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_ig_acceleration.pdf')
    fig.savefig(FIG_DIR / 'fig4_ig_acceleration.png')
    plt.close(fig)
    print("  Figure 4: fig4_ig_acceleration.pdf")


if __name__ == '__main__':
    print("Generating figures for Dream Layer v0.1 paper...")
    data = load_data()

    figure1_connections_per_concept(data)
    figure2_cumulative_ig(data)
    figure3_strategy_breakdown(data)
    figure4_ig_acceleration()

    print("\nAll figures generated.")
