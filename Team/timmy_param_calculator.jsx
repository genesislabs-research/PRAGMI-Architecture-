import { useState, useMemo, useCallback } from "react";

const VOCAB_SIZE = 128256;
const COORDINATE_DIM = 64;
const T_DEFAULT = 8;
const T_SLOW_DEFAULT = 2;
const BUFFER_DIM = 32;
const ROUTING_RANK = 64;
const WM_HIDDEN = 128;
const KERNEL_PARAMS = 700000;

function countColumn(cfg) {
  const {
    d_model, d_ff, n_experts, sensory_layers, assoc_layers, exec_layers,
    memory_size, n_clusters, T, T_slow,
  } = cfg;
  const T_total = T + T_slow;
  const n_heads = Math.floor(d_model / 64);

  const breakdown = {};

  // Encoder
  breakdown.embed = VOCAB_SIZE * d_model;
  breakdown.temporal_proj = d_model * d_model;
  breakdown.basis = 2 + T * d_model + T_slow * d_model;
  breakdown.float_inject = d_model * d_model + 1;

  // Input LIF
  breakdown.input_lif = 2 * d_model;

  // Attention (shared structure)
  const attn = 4 * d_model * d_model + 4 * d_model + 1 + 2 * T_total;
  const overhead = 4 * d_model + 2 + 1;
  const ffn = 2 * d_model * d_ff + 2 * d_ff + 2 * d_model;
  const expert_ff = Math.floor(d_ff / n_experts);
  const moe = n_experts * 2 * d_model * expert_ff + 2 * expert_ff + 2 * d_model + n_clusters * n_experts;

  breakdown.sensory = sensory_layers * (attn + overhead + ffn);
  breakdown.association = assoc_layers * (attn + overhead + moe);
  breakdown.executive = exec_layers * (attn + overhead + ffn);

  // Memory cortex
  breakdown.memory_cortex =
    3 * d_model * memory_size +
    memory_size * memory_size +
    4 * memory_size +
    memory_size +
    2 * d_model + 2;

  // Readout
  breakdown.readout = 2 * d_model + 1 + 2 * d_model;

  // Per-layer detail for display
  const attn_per_layer = attn;
  const ffn_per_layer = ffn;
  const moe_per_layer = moe;

  const total = Object.values(breakdown).reduce((a, b) => a + b, 0);

  return { breakdown, total, attn_per_layer, ffn_per_layer, moe_per_layer, n_heads };
}

function countShared(cfg) {
  const { d_model, n_cols } = cfg;
  const parts = {};

  parts.lm_head = COORDINATE_DIM * VOCAB_SIZE;
  parts.router = d_model * ROUTING_RANK + ROUTING_RANK * n_cols;
  parts.bridge = n_cols * (d_model * ROUTING_RANK + ROUTING_RANK + ROUTING_RANK * COORDINATE_DIM);
  parts.cortical_buffers = n_cols * 2 * d_model * BUFFER_DIM;
  parts.world_model = 5 * (COORDINATE_DIM * WM_HIDDEN + WM_HIDDEN * WM_HIDDEN + WM_HIDDEN * COORDINATE_DIM);
  parts.cognitive_kernel = KERNEL_PARAMS;

  const total = Object.values(parts).reduce((a, b) => a + b, 0);
  return { parts, total };
}

function fmt(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(3) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toString();
}

function fmtGB(bytes) {
  return (bytes / 1e9).toFixed(1) + " GB";
}

function Slider({ label, value, onChange, min, max, step, suffix, note }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
        <label style={{ fontSize: 13, color: "#c4c9d4", fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.3 }}>
          {label}
        </label>
        <span style={{
          fontSize: 15, fontWeight: 700, color: "#e8ecf4",
          fontFamily: "'JetBrains Mono', monospace",
          background: "rgba(99,132,255,0.10)", padding: "2px 8px", borderRadius: 4
        }}>
          {value.toLocaleString()}{suffix || ""}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#6384ff", cursor: "pointer" }}
      />
      {note && <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2, fontStyle: "italic" }}>{note}</div>}
    </div>
  );
}

function StatBox({ label, value, sub, color, wide }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 8, padding: "12px 14px",
      flex: wide ? "1 1 100%" : "1 1 calc(50% - 6px)",
      minWidth: wide ? "100%" : 140,
    }}>
      <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4, fontFamily: "'JetBrains Mono', monospace" }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 800, color: color || "#e8ecf4", fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: "#6b7280", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function BreakdownBar({ label, value, total, color }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#9ca3af", fontFamily: "'JetBrains Mono', monospace", marginBottom: 2 }}>
        <span>{label}</span>
        <span>{fmt(value)} ({pct.toFixed(1)}%)</span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.3s ease" }} />
      </div>
    </div>
  );
}

export default function TimmyParamCalculator() {
  const [d_model, setDModel] = useState(2368);
  const [d_ff_mult, setDFFMult] = useState(3);
  const [n_experts, setNExperts] = useState(8);
  const [sensory_layers, setSensory] = useState(8);
  const [assoc_layers, setAssoc] = useState(8);
  const [exec_layers, setExec] = useState(8);
  const [memory_size, setMemSize] = useState(256);
  const [n_clusters, setNClusters] = useState(64);
  const [n_cols, setNCols] = useState(6);
  const [T, setT] = useState(T_DEFAULT);
  const [T_slow, setTSlow] = useState(T_SLOW_DEFAULT);

  const d_ff = d_model * d_ff_mult;
  const n_heads = Math.floor(d_model / 64);
  const total_layers = sensory_layers + assoc_layers + exec_layers;

  const results = useMemo(() => {
    const colCfg = {
      d_model, d_ff, n_experts, sensory_layers, assoc_layers, exec_layers,
      memory_size, n_clusters, T, T_slow,
    };
    const col = countColumn(colCfg);
    const shared = countShared({ d_model, n_cols });
    const grand = col.total * n_cols + shared.total;
    const embedFraction = (col.breakdown.embed * n_cols) / grand * 100;

    // VRAM estimates
    const weightsFp16 = grand * 2;
    const adamFp32 = grand * 8;
    const gradsFp16 = grand * 2;
    const minVRAM = weightsFp16 + adamFp32 + gradsFp16;

    return { col, shared, grand, embedFraction, weightsFp16, adamFp32, gradsFp16, minVRAM };
  }, [d_model, d_ff, n_experts, sensory_layers, assoc_layers, exec_layers, memory_size, n_clusters, n_cols, T, T_slow]);

  const { col, shared, grand, embedFraction } = results;
  const diffFrom10B = ((grand - 10e9) / 10e9 * 100);

  const targetColor = Math.abs(diffFrom10B) < 1 ? "#4ade80" :
                       Math.abs(diffFrom10B) < 5 ? "#facc15" : "#f87171";

  return (
    <div style={{
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      background: "#0c0e14",
      color: "#e8ecf4",
      minHeight: "100vh",
      padding: "24px 16px",
    }}>
      <div style={{ maxWidth: 680, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: targetColor,
              boxShadow: `0 0 8px ${targetColor}40`,
            }} />
            <h1 style={{
              fontSize: 20, fontWeight: 800, margin: 0,
              fontFamily: "'JetBrains Mono', monospace",
              background: "linear-gradient(135deg, #6384ff, #a78bfa)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              PRAGMI Parameter Calculator
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "#6b7280", margin: 0, paddingLeft: 18 }}>
            Timmy v2 multi-column SNN architecture with shared LM head at 64-dim coordinate projection
          </p>
        </div>

        {/* Grand Total */}
        <div style={{
          background: `linear-gradient(135deg, ${targetColor}08, ${targetColor}03)`,
          border: `1px solid ${targetColor}25`,
          borderRadius: 12, padding: "16px 18px", marginBottom: 20,
          display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8,
        }}>
          <div>
            <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
              Grand Total
            </div>
            <div style={{ fontSize: 32, fontWeight: 900, color: "#e8ecf4", fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.1 }}>
              {(grand / 1e9).toFixed(3)}B
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 13, color: targetColor, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
              {diffFrom10B >= 0 ? "+" : ""}{diffFrom10B.toFixed(2)}% from 10B
            </div>
            <div style={{ fontSize: 11, color: "#6b7280" }}>
              {n_cols} columns, {total_layers} layers, {n_heads} heads
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
          <StatBox label="Per Column" value={fmt(col.total)} sub={`${(col.total / grand * 100).toFixed(1)}% of total`} />
          <StatBox label="Shared Infra" value={fmt(shared.total)} sub="LM head + bridge + kernel" />
          <StatBox label="Embed Fraction" value={embedFraction.toFixed(1) + "%"} sub={`${fmt(col.breakdown.embed)} per column`} color={embedFraction > 25 ? "#facc15" : "#4ade80"} />
          <StatBox label="Min VRAM" value={fmtGB(results.minVRAM)} sub="fp16 weights + fp32 Adam + grads" color={results.minVRAM > 160e9 ? "#f87171" : results.minVRAM > 80e9 ? "#facc15" : "#4ade80"} />
        </div>

        {/* Controls */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12, padding: "18px 16px", marginBottom: 20,
        }}>
          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "0 0 14px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            Column Dimensions
          </h2>
          <Slider label="d_model" value={d_model} onChange={setDModel} min={256} max={4096} step={64} note={`head_dim=64, ${n_heads} heads. Must be multiple of 64.`} />
          <Slider label="d_ff multiplier" value={d_ff_mult} onChange={setDFFMult} min={2} max={6} step={1} suffix={"x"} note={`d_ff = ${d_ff.toLocaleString()}`} />
          <Slider label="n_experts (MoE)" value={n_experts} onChange={setNExperts} min={2} max={32} step={2} note={`expert_ff = ${Math.floor(d_ff / n_experts).toLocaleString()} per expert`} />

          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "18px 0 14px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            Zone Depth
          </h2>
          <Slider label="Sensory layers (FFN)" value={sensory_layers} onChange={setSensory} min={1} max={16} step={1} />
          <Slider label="Association layers (MoE)" value={assoc_layers} onChange={setAssoc} min={1} max={16} step={1} />
          <Slider label="Executive layers (FFN)" value={exec_layers} onChange={setExec} min={1} max={16} step={1} />

          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "18px 0 14px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            System
          </h2>
          <Slider label="Columns (Prime + specialists)" value={n_cols} onChange={setNCols} min={1} max={16} step={1} />
          <Slider label="Memory cortex size" value={memory_size} onChange={setMemSize} min={64} max={1024} step={64} />
          <Slider label="Minicolumn clusters" value={n_clusters} onChange={setNClusters} min={16} max={256} step={16} />
          <div style={{ display: "flex", gap: 12 }}>
            <div style={{ flex: 1 }}>
              <Slider label="T (gamma)" value={T} onChange={setT} min={4} max={16} step={1} />
            </div>
            <div style={{ flex: 1 }}>
              <Slider label="T_slow (theta)" value={T_slow} onChange={setTSlow} min={1} max={4} step={1} />
            </div>
          </div>
        </div>

        {/* Breakdown */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12, padding: "18px 16px", marginBottom: 20,
        }}>
          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "0 0 14px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            Per-Column Breakdown
          </h2>
          <BreakdownBar label="Embedding table" value={col.breakdown.embed} total={col.total} color="#6384ff" />
          <BreakdownBar label={`Sensory zone (${sensory_layers} FFN)`} value={col.breakdown.sensory} total={col.total} color="#4ade80" />
          <BreakdownBar label={`Association zone (${assoc_layers} MoE)`} value={col.breakdown.association} total={col.total} color="#facc15" />
          <BreakdownBar label={`Executive zone (${exec_layers} FFN)`} value={col.breakdown.executive} total={col.total} color="#f87171" />
          <BreakdownBar label="Memory cortex" value={col.breakdown.memory_cortex} total={col.total} color="#a78bfa" />
          <BreakdownBar label="Encoder (proj + basis)" value={col.breakdown.temporal_proj + col.breakdown.basis + col.breakdown.float_inject} total={col.total} color="#38bdf8" />
          <BreakdownBar label="LIF populations + readout" value={col.breakdown.input_lif + col.breakdown.readout} total={col.total} color="#818cf8" />

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", marginTop: 12, paddingTop: 12 }}>
            <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "0 0 10px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
              Per-Layer Detail
            </h2>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 140 }}>
                <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>Attention</div>
                <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{fmt(col.attn_per_layer)}</div>
              </div>
              <div style={{ flex: 1, minWidth: 140 }}>
                <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>FFN</div>
                <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{fmt(col.ffn_per_layer)}</div>
              </div>
              <div style={{ flex: 1, minWidth: 140 }}>
                <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>MoE</div>
                <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{fmt(col.moe_per_layer)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* VRAM */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12, padding: "18px 16px", marginBottom: 20,
        }}>
          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "0 0 14px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            VRAM Estimate (fp16 training, Adam)
          </h2>
          <BreakdownBar label="Weights (fp16)" value={results.weightsFp16} total={results.minVRAM} color="#6384ff" />
          <BreakdownBar label="Adam states (fp32 momentum + variance)" value={results.adamFp32} total={results.minVRAM} color="#facc15" />
          <BreakdownBar label="Gradients (fp16)" value={results.gradsFp16} total={results.minVRAM} color="#4ade80" />
          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 8, fontStyle: "italic" }}>
            Excludes activation memory (depends on batch size, seq len, gradient checkpointing).
            With gradient checkpointing enabled, activation memory is roughly O(sqrt(layers)).
          </div>
        </div>

        {/* Architecture constants */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12, padding: "14px 16px", marginBottom: 20,
        }}>
          <h2 style={{ fontSize: 13, fontWeight: 700, color: "#6384ff", margin: "0 0 10px 0", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
            Fixed Architecture Constants
          </h2>
          <div style={{ fontSize: 12, color: "#9ca3af", fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8 }}>
            vocab_size = {VOCAB_SIZE.toLocaleString()} (Llama-3.2-1B tokenizer)<br/>
            coordinate_dim = {COORDINATE_DIM} (Perforant Path manifold)<br/>
            head_dim = 64 (n_heads = d_model / 64)<br/>
            buffer_dim = {BUFFER_DIM} (cortical buffer bottleneck)<br/>
            routing_rank = {ROUTING_RANK} (column router subspace)<br/>
            LM head = shared at coordinate_dim ({fmt(COORDINATE_DIM * VOCAB_SIZE)} params)<br/>
            Cognitive kernel ~ {fmt(KERNEL_PARAMS)} (hippocampal memory system)
          </div>
        </div>

        <div style={{ textAlign: "center", fontSize: 11, color: "#374151", padding: "8px 0 16px" }}>
          Genesis Labs Research | PRAGMI Architecture
        </div>
      </div>
    </div>
  );
}
