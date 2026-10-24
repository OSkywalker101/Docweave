#!/usr/bin/env python3
"""Generate static chart images for the comparison paper (no JS required)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'docweave': '#2E86AB', 'markitdown': '#A23B72', 'tie': '#6C757D'}

# 1. Accuracy Comparison - Grouped Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DocWeave vs MarkItDown: Measured Accuracy (22 Documents, Ground Truth)', fontsize=14, fontweight='bold')

# Word-level metrics
ax = axes[0, 0]
metrics = ['Word\nRecall', 'Word\nPrecision', 'Word\nF1', 'Reading\nOrder']
dw = [0.9432, 0.8433, 0.8656, 0.8206]
mid = [0.9126, 0.8004, 0.8310, 0.5892]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, dw, width, label='DocWeave', color=COLORS['docweave'], edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, mid, width, label='MarkItDown', color=COLORS['markitdown'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score (higher = better)')
ax.set_title('Word-Level & Reading Order Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
for i, (d, m) in enumerate(zip(dw, mid)):
    ax.text(i - width/2, d + 0.02, f'{d:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, m + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

# Table metrics
ax = axes[0, 1]
metrics2 = ['Cell\nRecall', 'Cell\nPrecision']
dw2 = [1.0000, 1.0000]
mid2 = [0.9381, 0.8582]
x2 = np.arange(len(metrics2))
ax.bar(x2 - width/2, dw2, width, label='DocWeave', color=COLORS['docweave'], edgecolor='white', linewidth=0.5)
ax.bar(x2 + width/2, mid2, width, label='MarkItDown', color=COLORS['markitdown'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score (higher = better)')
ax.set_title('Table Cell Accuracy (12 Documents)')
ax.set_xticks(x2)
ax.set_xticklabels(metrics2)
ax.set_ylim(0, 1.05)
ax.legend()
for i, (d, m) in enumerate(zip(dw2, mid2)):
    ax.text(i - width/2, d + 0.02, f'{d:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, m + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

# CER (lower is better)
ax = axes[1, 0]
metrics3 = ['CER\n(lower=better)']
dw3 = [0.2107]
mid3 = [0.7490]
x3 = np.arange(len(metrics3))
ax.bar(x3 - width/2, dw3, width, label='DocWeave', color=COLORS['docweave'], edgecolor='white', linewidth=0.5)
ax.bar(x3 + width/2, mid3, width, label='MarkItDown', color=COLORS['markitdown'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Character Error Rate')
ax.set_title('Character Error Rate')
ax.set_xticks(x3)
ax.set_xticklabels(metrics3)
ax.set_ylim(0, 1.0)
ax.legend()
for i, (d, m) in enumerate(zip(dw3, mid3)):
    ax.text(i - width/2, d + 0.02, f'{d:.4f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, m + 0.02, f'{m:.4f}', ha='center', va='bottom', fontsize=8)

# Diagram metrics (DocWeave only)
ax = axes[1, 1]
diag_metrics = ['Kind\nAccuracy', 'Node\nRecall', 'Node\nPrecision', 'Edge\nRecall', 'Edge\nPrecision', 'Direction\nProof']
diag_values = [1.000, 1.000, 1.000, 0.8788, 1.000, 0.787]
bars = ax.bar(diag_metrics, diag_values, color=COLORS['docweave'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score')
ax.set_title('DocWeave Diagram Reconstruction (11 Diagrams)')
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, diag_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('E:\\gitprac1\\markmedown\\docs\\accuracy_charts.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. Latency Comparison
fig, ax = plt.subplots(figsize=(12, 6))
files = ['bad.xml', 'bom.txt', 'ctrl_chars.pdf', 'diagram.docx', 'empty.pptx', 'empty_pages.pdf',
         'formatted.xlsx', 'hostile.xlsx', 'image.docx', 'image.pptx', 'large.csv', 'liar.pdf',
         'multicolumn.pdf', 'notes.docx', 'order.docx', 'ragged.csv', 'real_scan.pdf', 'sample.csv',
         'sample.docx', 'sample.html', 'sample.json', 'sample.md', 'sample.pptx', 'sample.xlsx',
         'sample.xml', 'scanned.pdf', 'simple.pdf', 'sparse.xlsx', 'table.pdf']
dw_lat = [1.9, 2.5, 7.3, 44.7, 21.4, 6.4, 28.2, 16.6, 35.2, 22.5, 473.2, 1.7, 15.4, 7.5, 33.8, 2.0, 158.2, 2.1, 45.0, 13.3, 2.3, 3.3, 50.5, 35.1, 1.8, 7.6, 18.8, 15.5, 26.6]
mid_lat = [2.1, 2.5, 8.2, 44.7, 21.4, 7.3, 29.4, 16.6, 35.3, 24.1, 575.3, 1.8, 19.5, 10.6, 40.3, 2.6, 158.2, 2.4, 45.0, 13.3, 2.5, 3.3, 51.2, 36.9, 1.8, 9.6, 20.4, 15.5, 29.0]

x = np.arange(len(files))
ax.bar(x - width/2, dw_lat, width, label='DocWeave', color=COLORS['docweave'], alpha=0.8)
ax.bar(x + width/2, mid_lat, width, label='MarkItDown', color=COLORS['markitdown'], alpha=0.8)
ax.set_ylabel('Latency (ms)')
ax.set_title('Per-File Median Latency (3 runs each)')
ax.set_xticks(x)
ax.set_xticklabels(files, rotation=90, fontsize=7)
ax.legend()
ax.set_yscale('log')
ax.set_ylim(0.5, 1000)
plt.tight_layout()
plt.savefig('E:\\gitprac1\\markmedown\\docs\\latency_chart.png', dpi=200, bbox_inches='tight')
plt.close()

# 3. Per-Document Word F1
fig, ax = plt.subplots(figsize=(12, 6))
doc_names = ['bom.txt', 'diagram.docx', 'formatted.xlsx', 'hostile.xlsx', 'image.docx',
             'multicolumn.pdf', 'notes.docx', 'order.docx', 'ragged.csv', 'real_scan.pdf',
             'sample.csv', 'sample.docx', 'sample.html', 'sample.json', 'sample.md',
             'sample.pptx', 'sample.xlsx', 'sample.xml', 'scanned.pdf', 'simple.pdf',
             'sparse.xlsx', 'table.pdf']
dw_f1 = [1.000, 1.000, 0.846, 0.941, 1.000, 1.000, 0.952, 0.857, 1.000, 0.000,
         1.000, 0.966, 0.974, 0.857, 0.960, 1.000, 0.833, 1.000, 0.000, 1.000, 0.857, 1.000]
mid_f1 = [1.000, 1.000, 0.756, 0.625, 1.000, 1.000, 0.714, 0.857, 0.957, 0.000,
          1.000, 0.966, 0.950, 0.783, 0.960, 0.983, 0.378, 1.000, 1.000, 1.000, 0.353, 1.000]

x = np.arange(len(doc_names))
ax.bar(x - width/2, dw_f1, width, label='DocWeave', color=COLORS['docweave'], edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, mid_f1, width, label='MarkItDown', color=COLORS['markitdown'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Word F1')
ax.set_title('Per-Document Word F1 (22 Documents with Ground Truth)')
ax.set_xticks(x)
ax.set_xticklabels(doc_names, rotation=90, fontsize=7)
ax.set_ylim(0, 1.05)
ax.legend()
# Mark scanned docs
for i, name in enumerate(doc_names):
    if 'scan' in name or 'real' in name:
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='gray')
plt.tight_layout()
plt.savefig('E:\\gitprac1\\markmedown\\docs\\per_doc_f1.png', dpi=200, bbox_inches='tight')
plt.close()

# 4. Diagram Detail - Edge Recall by Diagram
fig, ax = plt.subplots(figsize=(10, 5))
diag_names = ['ambiguous', 'architecture', 'box_and_arrow', 'er', 'flowchart',
              'grouped_subsystem', 'network_topology', 'parallelogram', 'sequence',
              'state_machine', 'uml_class']
edge_recall = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6667, 1.0, 1.0]
dir_proof = [1.0, 0.8, 1.0, 0.0, 0.857, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5]
xfail = [True, False, False, False, False, False, False, False, True, False, False]

bars1 = ax.bar(np.arange(len(diag_names)) - 0.2, edge_recall, 0.4, label='Edge Recall', color=COLORS['docweave'], alpha=0.8)
bars2 = ax.bar(np.arange(len(diag_names)) + 0.2, dir_proof, 0.4, label='Direction Proof Rate', color='#F18F01', alpha=0.8)

for i, (b1, b2, xf) in enumerate(zip(bars1, bars2, xfail)):
    if xf:
        b1.set_hatch('///')
        b2.set_hatch('///')

ax.set_ylabel('Score')
ax.set_title('DocWeave Diagram Engine: Edge Recall & Direction Proof (11 Diagrams)\n/// = Strict XFAIL (documented limitations)')
ax.set_xticks(np.arange(len(diag_names)))
ax.set_xticklabels(diag_names, rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig('E:\\gitprac1\\markmedown\\docs\\diagram_detail.png', dpi=200, bbox_inches='tight')
plt.close()

print('Charts generated: accuracy_charts.png, latency_chart.png, per_doc_f1.png, diagram_detail.png')
