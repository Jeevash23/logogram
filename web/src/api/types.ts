// Types mirroring the Python API payloads (src/logogram/server/app.py and friends).

export type ThemeSetting = "light" | "dark" | "system";

export interface ModelInfo {
  id: string;
  revision: string | null;
  architecture: string;
  n_layers: number;
  n_heads: number;
  d_model: number;
  d_head: number;
  d_mlp: number | null;
  d_vocab: number;
  n_ctx: number;
  n_params: number | null;
  dtype: Dtype;
  device: "cpu" | "cuda" | "mps";
  device_name: string;
  process_weights: boolean;
  site_kinds: string[];
  backend: string;
  backend_version: string;
  extra?: { block_structure?: string; normalization?: string; activation?: string; prediction_method?: string | null };
}

export interface ModelStatus {
  state: "none" | "loading" | "ready" | "error";
  id?: string;
  stage?: "resolving" | "downloading" | "loading" | "processing" | "ready";
  done?: number;
  total?: number;
  file?: string;
  error?: string;
  revision?: string;
  info?: ModelInfo;
  memory?: number | null;
  ref?: { id: string; revision: string | null; dtype: Dtype; device: Device; process_weights: boolean };
}

export interface Job {
  project_session?: string | null;
  id: string;
  kind: "load_model" | "run";
  title: string;
  status: "running" | "finished" | "failed" | "cancelled";
  run_id: string | null;
  progress: Record<string, unknown>;
  error: string | null;
  started: number;
  /** Increases with every change of status: a lower version is an older snapshot. */
  version: number;
  cancelling?: boolean;
}

export interface DatasetListing {
  name: string;
  path: string;
  modified: string | null;
  n?: number;
  has_positions?: boolean;
  sha256?: string;
  error?: string;
}

export interface ProjectInfo {
  session_id: string;
  name: string;
  description: string;
  path: string;
  datasets: DatasetListing[];
}

export interface ServerState {
  version: string;
  project: ProjectInfo | null;
  model: ModelStatus;
  job: Job | null;
  first_run: boolean;
  theme: ThemeSetting;
  projects_parent: string;
}

export interface RecentProject {
  path: string;
  name: string;
  opened: string | null;
  modified: string;
}

export interface FolderListing {
  path: string;
  parent: string | null;
  is_project: boolean;
  entries: { name: string; path: string; is_project: boolean }[];
}

export interface SystemIssue {
  severity: "error" | "warning" | "info";
  title: string;
  detail: string;
  fix: string | null;
}

export interface SystemReport {
  os: string;
  machine: string;
  python: string;
  cpu: string;
  cpu_cores: number | null;
  cpu_threads: number | null;
  memory_total: number;
  memory_available: number;
  gpus: { name: string; memory_total: number; memory_free: number | null; bf16: boolean }[];
  backend: "cuda" | "mps" | "cpu";
  torch: string;
  torch_cuda: string | null;
  transformer_lens: string | null;
  transformers: string | null;
  recommended_dtype: Dtype;
  precision_note: string;
  hf_cache_free: number | null;
  issues: SystemIssue[];
}

// -- spec --------------------------------------------------------------------------------------

export type Dtype = "float32" | "float16" | "bfloat16";
export type Device = "auto" | "cpu" | "cuda" | "mps";
export type StreamKind = "resid_pre" | "resid_mid" | "resid_post" | "attn_out" | "mlp_out";
export type SiteKind = StreamKind | "head";

export type PositionSpec =
  | { kind: "all" }
  | { kind: "last" }
  | { kind: "index"; index: number }
  | { kind: "label"; label: string };

export interface SiteSpec {
  kind: SiteKind;
  layer: number;
  head?: number | null;
  position: PositionSpec;
}

export type ScopeSpec =
  | { kind: "heads"; position: PositionSpec }
  | { kind: "layer_position"; site: StreamKind; positions: "each" | "labels" }
  | { kind: "layer_components"; components: StreamKind[]; position: PositionSpec }
  | { kind: "sites"; sites: SiteSpec[] };

export type BaselineSpec =
  | { kind: "zero" }
  | { kind: "mean"; reference: "clean" | "corrupt" }
  | { kind: "resample"; pool: "clean" | "corrupt"; donors: number; seed: number };

export type ExperimentSpec =
  | { kind: "activation_patching"; direction: "clean_to_corrupt" | "corrupt_to_clean" }
  | { kind: "ablation"; baseline: BaselineSpec };

export interface Spec {
  logogram_spec: 1;
  name: string;
  notes: string;
  model: {
    id: string;
    revision: string | null;
    dtype: Dtype;
    device: Device;
    process_weights: boolean;
  };
  dataset: { path: string; sha256: string | null; limit: number | null };
  tokenization: { prepend_bos: boolean };
  experiment: ExperimentSpec;
  scope: ScopeSpec;
  metric: { kind: "logit_diff"; normalization: "dataset_gap" | "prompt_gap" };
  statistics: { bootstrap: number; ci: number; seed: number };
  execution: { batch_size: number };
  predictions?: PredictionSettings | null;
}

export interface PredictionSettings {
  method: "final_norm_logit_lens";
  prompt_index: number;
  which: "clean" | "corrupt";
  position: { kind: "last" } | { kind: "index"; index: number };
  top_k: number;
}

export interface PredictionReport {
  index: number;
  position: number;
  tokens: string[];
  settings: PredictionSettings;
  layers: { layer: number; top: { id: number; token: string; prob: number }[]; answer_prob: number; distractor_prob: number; logit_diff: number }[];
  batch_members: number[];
  answer: string;
  distractor: string;
  issues: { index: number; kind: string; message: string }[];
}

export interface NoteInput {
  title: string;
  body: string;
  model: Spec["model"];
  sites: SiteSpec[];
  run_id: string | null;
}

export interface ResearchNote extends NoteInput {
  id: string;
  revision: number;
  created: string;
  updated: string;
}

// -- results -----------------------------------------------------------------------------------

export interface Stat {
  mean: number | null;
  sd: number | null;
  lo: number | null;
  hi: number | null;
}

export interface SiteBase {
  index: number;
  kind: SiteKind;
  layer: number;
  head: number | null;
  position: PositionSpec;
  position_key: string;
  row: number;
  col: number;
  label: string;
}

export interface SiteResult extends SiteBase {
  n: number;
  effect: Stat;
  delta: Stat;
  patched_logit_diff: number | null;
  answer_prob: number | null;
  answer_prob_delta: number | null;
  sign_flips: number;
  opposite_sign: number;
}

export interface LayoutAxis {
  key: string;
  label: string;
  position?: number;
  clean?: string;
  corrupt?: string;
  differs?: boolean;
}

export interface Layout {
  kind: "heads" | "layer_position" | "layer_components" | "sites";
  site?: StreamKind;
  row_title: string;
  col_title: string;
  rows: LayoutAxis[];
  cols: LayoutAxis[];
}

export interface GroupStats {
  mean: number | null;
  sd: number | null;
}

export interface ModelShape {
  id: string;
  n_layers: number;
  n_heads: number;
  d_model?: number;
  d_head?: number;
  d_mlp?: number | null;
  site_kinds?: string[] | null;
  block_structure?: string | null;
  normalization?: string | null;
  activation?: string | null;
}

export interface Summary {
  run_id: string;
  name: string;
  description: string;
  model: ModelShape | null;
  n_prompts: number;
  n_sites: number;
  layout: Layout;
  receiver: "clean" | "corrupt";
  reference: "clean" | "corrupt";
  metric: {
    kind: string;
    normalization: "dataset_gap" | "prompt_gap";
    denominator: number | null;
    description: string;
    normalized_effect: string;
  };
  statistics: { bootstrap: number; ci: number; seed: number; method: string };
  baseline: {
    clean: { logit_diff: GroupStats; answer_prob: GroupStats; prefers_answer: number };
    corrupt: { logit_diff: GroupStats; answer_prob: GroupStats; prefers_answer: number };
    gap: GroupStats;
  };
  sites: SiteResult[];
  per_prompt: Record<string, (number | null)[]>;
  donors: number[][] | null;
  warnings: string[];
}

export interface Manifest {
  run_id: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  wall_time_s?: number;
  error?: string | null;
  versions?: Record<string, string | null>;
  model?: {
    id: string;
    revision: string;
    architecture: string;
    process_weights: boolean;
    backend: string;
    backend_version: string;
  };
  device?: {
    type: string;
    name: string;
    cuda: string | null;
    threads: number;
    deterministic_algorithms: boolean;
  };
  dtype?: string;
  dataset?: { path: string; sha256: string; n: number };
  derived_from?: DerivedFrom | null;
}

export interface DerivedFrom {
  run: string;
  kind: "robustness" | "rerun";
  change?: string;
}

export interface RunListing {
  id: string;
  name: string;
  description: string;
  status: "finished" | "failed" | "cancelled" | "draft" | "running";
  created: string | null;
  finished: string | null;
  wall_time_s: number | null;
  n_prompts: number | null;
  layout_kind: Layout["kind"] | null;
  model_id: string;
  dataset: string;
  experiment: ExperimentSpec;
  scope: ScopeSpec;
  derived_from: DerivedFrom | null;
  error: string | null;
}

export interface RunDetail {
  id: string;
  spec: Spec;
  summary: Summary | null;
  manifest: Manifest | null;
  predictions?: PredictionReport | null;
  listing: RunListing | null;
  folder: string;
}

export interface PromptEvidence {
  index: number;
  clean: string | null;
  corrupt: string | null;
  answer: string | null;
  distractor: string | null;
  effect: number | null;
  delta: number | null;
  patched_logit_diff: number | null;
  receiver_logit_diff: number | null;
  reference_logit_diff: number | null;
  patched_answer_prob: number | null;
  receiver_answer_prob: number | null;
  flipped: boolean;
}

export interface SiteDetail {
  run_id: string;
  site: SiteResult;
  receiver: "clean" | "corrupt";
  reference: "clean" | "corrupt";
  metric: Summary["metric"];
  statistics: Summary["statistics"];
  prompts: PromptEvidence[];
  strongest: number[];
  weakest: number[];
  dataset_changed: boolean;
  dataset_available: boolean;
}

// -- prompts and analyses ----------------------------------------------------------------------

export interface PromptRecord {
  clean: string;
  corrupt: string;
  answer: string;
  distractor: string;
  positions?: Record<string, [number, number]>;
  id?: string;
  meta?: Record<string, unknown>;
}

export interface PromptIssue {
  index: number;
  kind: string;
  message: string;
}

export interface DatasetDetail {
  name: string;
  path: string;
  sha256: string;
  n: number;
  records: PromptRecord[];
  issues?: PromptIssue[];
  lengths?: number[];
}

export interface TokenStripData {
  clean: { tokens: string[]; ids: number[] };
  corrupt: { tokens: string[]; ids: number[] };
  aligned: boolean;
  differs: number[];
  labels: Record<string, number>;
  answer: { text: string; tokens: string[]; id: number | null };
  distractor: { text: string; tokens: string[]; id: number | null };
  issues: PromptIssue[];
}

export interface TopToken {
  token: string;
  id: number;
  prob: number;
}

export interface BaselinePrompt {
  index: number;
  clean: string;
  corrupt: string;
  answer: string;
  distractor: string;
  clean_logit_diff: number | null;
  corrupt_logit_diff: number | null;
  clean_answer_prob: number | null;
  corrupt_answer_prob: number | null;
  clean_top: TopToken[];
  corrupt_top: TopToken[];
}

export interface BaselineReport {
  n: number;
  prompts: BaselinePrompt[];
  issues: PromptIssue[];
  summary: {
    clean_logit_diff: number | null;
    corrupt_logit_diff: number | null;
    gap: number | null;
    clean_prefers_answer: number;
    corrupt_prefers_answer: number;
    clean_answer_prob: number | null;
    corrupt_answer_prob: number | null;
  } | null;
}

export interface AttentionData {
  layer: number;
  head: number;
  which: "clean" | "corrupt";
  index: number;
  tokens: string[];
  labels: Record<string, number>;
  pattern: number[][];
  average: number[][];
  /** Tokens shared by every averaged prompt (null where they differ). */
  average_tokens: (string | null)[];
  /** Labels at the same position in every averaged prompt. */
  average_labels: Record<string, number>;
  n_average: number;
  n_total: number;
  length: number;
}

export interface ComparisonChange {
  label: string;
  kind: SiteKind;
  layer: number;
  head: number | null;
  position_key: string;
  index_a: number;
  index_b: number;
  row: number;
  col: number;
  effect_a: Stat;
  effect_b: Stat;
  rank_a: number;
  rank_b: number;
  flags: ("sign" | "left_top" | "entered_top")[];
}

export interface Comparison {
  run_a: string;
  run_b: string;
  n_common: number;
  n_a: number;
  n_b: number;
  spearman: number | null;
  top_k: number;
  top_overlap: number;
  top_a: string[];
  top_b: string[];
  changes: ComparisonChange[];
  flagged: number[];
  n_sign_changes: number;
  diff: { index_a: number; row: number; col: number; value: number }[];
  same_layout: boolean;
  spec_differences: { path: string; a: unknown; b: unknown }[];
}

export interface MemoryEstimate {
  device: string;
  dtype: Dtype;
  n_params: number;
  weights: number;
  activations: number;
  margin: number;
  total: number;
  available: number;
  verdict: "fits" | "tight" | "wont_fit";
  explanation: string;
}

export interface EstimateResponse {
  id: string;
  revision: string;
  architecture: {
    n_layers: number;
    n_heads: number;
    d_model: number;
    d_mlp: number;
    d_vocab: number;
    n_ctx: number;
    architecture: string;
  };
  download_bytes: number;
  total_bytes: number;
  gated: boolean;
  estimate: MemoryEstimate;
}

export interface IOITemplate {
  id: string;
  text: string;
  default: boolean;
}
