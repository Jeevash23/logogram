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
  extra?: {
    /** How each layer adds attention and MLP into the residual stream, as measured on loading. */
    block_structure?: string;
    normalization?: string;
    activation?: string;
    prediction_method?: string | null;
    model_type?: string;
    n_key_value_heads?: number;
    /** Whether the tokenizer has a beginning-of-sequence token to prepend. */
    bos?: boolean;
    checks?: ModelChecks;
  };
}

/** What Logogram measured about a model when it loaded (backends/transformer_lens.check_model). */
export interface ModelChecks {
  tolerance: number;
  function: number;
  structure: string;
  residual: number | null;
  lens: number | null;
}

export interface ModelPreset {
  id: string;
  label: string;
  detail: string;
  tested: boolean;
  gated: boolean;
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
  kind: "load_model" | "load_sae" | "run";
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

/** What is known about newer versions of Logogram (see src/logogram/updates.py). */
export interface UpdateStatus {
  current: string;
  released: string | null;
  /** The user's choice about daily checks; null until asked. */
  automatic: boolean | null;
  latest: string | null;
  available: boolean;
  /** This version is months old and nothing newer is known: worth a look. */
  old: boolean;
  checked_at: string | null;
  error: string | null;
  notes_url: string | null;
  command: string;
}

export interface ServerState {
  version: string;
  project: ProjectInfo | null;
  model: ModelStatus;
  job: Job | null;
  first_run: boolean;
  theme: ThemeSetting;
  projects_parent: string;
  update: UpdateStatus;
  sae?: SAEStatus;
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
  /** Apple's GPU is there, but used only when chosen (TransformerLens reports silent errors on it). */
  mps_available: boolean;
  issues: SystemIssue[];
}

// -- spec --------------------------------------------------------------------------------------

export type Dtype = "float32" | "float16" | "bfloat16";
/** A robustness check reruns a sweep with one choice changed: the experiment, or the precision. */
export type RobustnessChange = { experiment: ExperimentSpec } | { dtype: Dtype };
export type Device = "auto" | "cpu" | "cuda" | "mps";
export type StreamKind = "resid_pre" | "resid_mid" | "resid_post" | "attn_out" | "mlp_out";
export type SiteKind = StreamKind | "head" | "sae_feature";

export type PositionSpec =
  | { kind: "all" }
  | { kind: "last" }
  | { kind: "index"; index: number }
  | { kind: "label"; label: string };

export interface SiteSpec {
  kind: SiteKind;
  layer: number;
  head?: number | null;
  /** A feature of the spec's SAE (kind sae_feature). */
  feature?: number | null;
  position: PositionSpec;
}

/** A published sparse autoencoder: repository, folder and exact commit. */
export interface SAERef {
  repo: string;
  path: string;
  revision: string | null;
}

export interface SAEFit {
  variance_explained: number;
  l0: number;
  tokens: number;
  logit_diff?: number | null;
  spliced_logit_diff?: number | null;
  n?: number;
  skipped?: number;
}

export interface SAEInfo extends SAERef {
  site: StreamKind;
  layer: number;
  d_in: number;
  d_sae: number;
  activation: string;
  k: number | null;
  normalize: string;
  format: string;
  note: string;
  fit: SAEFit | null;
}

export interface SAEStatus {
  state: "none" | "loading" | "ready" | "error";
  repo?: string;
  path?: string;
  stage?: string;
  done?: number;
  total?: number;
  error?: string;
  info?: SAEInfo;
}

export interface TokenFeatures {
  index: number;
  which: "clean" | "corrupt";
  tokens: string[];
  labels: Record<string, number>;
  features: { feature: number; activation: number }[][];
  active: number[];
  first_real_token: number;
}

export interface FeatureReport {
  feature: number;
  which: "clean" | "corrupt";
  index: number;
  tokens: string[];
  activations: number[];
  top: { index: number; max: number; position: number; token: string; text: string }[];
  active_prompts: number;
  n: number;
}

/** What "the rest of the model" is made of, for a set that keeps its sites and replaces the rest. */
export type UniverseKind = "head" | "attn_out" | "mlp_out";

/** Sites intervened on together, in one forward pass. With `complement`, the intervention covers
 * every component of the scope's universe except these sites ("keep the circuit, replace the
 * rest"); a complement set without sites replaces the whole universe. */
export interface SiteSetSpec {
  label: string;
  sites: SiteSpec[];
  complement: boolean;
}

export type ScopeSpec =
  | { kind: "heads"; position: PositionSpec }
  | { kind: "layer_position"; site: StreamKind; positions: "each" | "labels" }
  | { kind: "layer_components"; components: StreamKind[]; position: PositionSpec }
  | { kind: "sites"; sites: SiteSpec[] }
  /** Every feature of the spec's SAE; attribution patching keeps the top ones. With `choose_on`, a
   * seeded share of the prompts chooses them and the rest report them (both null: every prompt). */
  | { kind: "features"; position: PositionSpec; top: number; choose_on: number | null; seed: number | null }
  /** Sets of sites, each intervened on at once: one result per set (1 to 64 sets). */
  | { kind: "site_sets"; universe: UniverseKind[] | null; sets: SiteSetSpec[] };

export type BaselineSpec =
  | { kind: "zero" }
  | { kind: "mean"; reference: "clean" | "corrupt" }
  | { kind: "resample"; pool: "clean" | "corrupt"; donors: number; seed: number };

export type Direction = "clean_to_corrupt" | "corrupt_to_clean";

/** Where a path ends: a later head's query, key or value, or the logits read directly. */
export type PathReceiverSpec = { kind: "head"; layer: number; head: number; input: "q" | "k" | "v" } | { kind: "logits" };

/** How attribution patching estimates: one gradient at the receiver run, or integrated gradients
 * averaged over `steps` runs between the two prompts (2 to 64; null for one gradient). */
export type AttributionMethod = "gradient" | "integrated_gradients";

export type ExperimentSpec =
  | { kind: "activation_patching"; direction: Direction }
  | { kind: "ablation"; baseline: BaselineSpec }
  | { kind: "direct_logit_attribution"; prompts: "clean" | "corrupt" }
  | { kind: "attribution_patching"; direction: Direction; method: AttributionMethod; steps: number | null }
  | { kind: "path_patching"; direction: Direction; receivers: PathReceiverSpec[]; freeze_mlps: boolean }
  | {
      kind: "steering";
      apply_to: "clean" | "corrupt";
      coefficients: number[];
      train_fraction: number;
      seed: number;
      control: boolean;
    };

export type ExperimentKind = ExperimentSpec["kind"];

/** What a run's per-prompt values are: patched runs, estimates of them, or terms of a split. */
export type Measure = "intervention" | "estimate" | "attribution";

/** What a forward pass is measured by (src/logogram/spec.py, METRIC_LABELS). */
export type MetricKind = "logit_diff" | "logprob_diff" | "logprob" | "prob" | "prob_diff" | "kl";
export type Normalization = "dataset_gap" | "prompt_gap";
export type Side = "clean" | "corrupt";

/** The metric of a spec. The KL divergence is measured from a target prompt's next-token
 * distribution, which has no default. */
export type MetricSpec =
  | { kind: Exclude<MetricKind, "kl">; normalization: Normalization }
  | { kind: "kl"; target: Side; normalization: Normalization };

/** A version 2 spec: every choice that can change a number is stated. */
export interface Spec {
  logogram_spec: 2;
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
  metric: MetricSpec;
  /** Percentile bootstrap; with `cluster`, over groups of prompts sharing that field of their meta. */
  statistics: { bootstrap: number; ci: number; seed: number; cluster: string | null };
  execution: { batch_size: number };
  predictions?: PredictionSettings | null;
  sae?: SAERef | null;
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
  /** How the lens reads the answer: one token, a set (summed), or a continuation's first token. */
  answer_reading?: "token" | "set" | "first_token";
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

/** A site of a run's results: a site of the model, or a set of sites (a row of a site_sets run,
 * whose layer is -1 and whose members are its sites). */
export type ResultKind = SiteKind | "site_set";

export interface SiteBase {
  index: number;
  kind: ResultKind;
  layer: number;
  head: number | null;
  feature?: number | null;
  position: PositionSpec;
  position_key: string;
  row: number;
  col: number;
  label: string;
  /** One of several measurements of the same site, such as a steering strength or its control,
   * or a set of sites (whether it keeps its sites, and how many). */
  variant?: { coefficient?: number; control?: boolean; complement?: boolean; size?: number } | null;
  variant_key?: string | null;
  /** The sites of a set intervened on together (kind site_set). */
  members?: SiteSpec[] | null;
}

export interface SiteResult extends SiteBase {
  n: number;
  effect: Stat;
  delta: Stat;
  /** The spec's metric, patched (absent in runs that measured the logit difference only). */
  patched_metric?: number | null;
  /** log P(answer) − log P(distractor), patched: the logit difference for single tokens. */
  patched_logit_diff: number | null;
  answer_prob: number | null;
  answer_prob_delta: number | null;
  sign_flips: number;
  opposite_sign: number;
  /** Corrected for the number of sites (finished runs only): a band that holds for every site
   * together, and the Benjamini–Hochberg q-value. */
  band?: { lo: number; hi: number } | null;
  q?: number | null;
}

export interface LayoutAxis {
  key: string;
  label: string;
  position?: number;
  clean?: string;
  corrupt?: string;
  differs?: boolean;
  /** Steering columns: the strength, and whether it is along the random control direction. */
  coefficient?: number;
  control?: boolean;
  /** Sets of sites: whether the row's set keeps its sites, and how many it holds. */
  complement?: boolean;
  size?: number;
}

export interface Layout {
  kind: "heads" | "layer_position" | "layer_components" | "sites" | "steering" | "site_sets";
  site?: StreamKind;
  row_title: string;
  col_title: string;
  rows: LayoutAxis[];
  cols: LayoutAxis[];
  /** Sets of sites: the components a set that keeps its sites replaces. */
  universe?: UniverseKind[] | null;
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
  /** Absent in runs from before it existed: those are interventions. */
  measure?: Measure;
  metric: SummaryMetric;
  statistics: SummaryStatistics;
  baseline: {
    clean: PromptSetStats;
    corrupt: PromptSetStats;
    /** The metric's gap the effects are normalized by: reference − receiver. */
    gap: GroupStats;
  };
  sites: SiteResult[];
  per_prompt: Record<string, (number | null)[]>;
  donors: number[][] | null;
  warnings: string[];
  /** Direct logit attribution: the mean logit difference and what it splits into. */
  direct?: DirectSplit | null;
  /** Steering: the prompts that trained the directions, the held-out ones measured, the length of
   * each row's direction, and each steered site and strength against its random control. */
  steering?: { train: number[]; test: number[]; norms: number[]; control?: ControlComparison[] | null } | null;
  /** Runs on SAE features: the SAE, its fit on these prompts, and (attribution patching) how much
   * of the site's estimated effect its features account for. */
  features?: {
    sae: SAEInfo;
    fit: SAEFit;
    site_estimate?: number | null;
    features_estimate?: number | null;
    evaluated?: number | null;
    /** Every feature, chosen on some prompts and reported on the others: their dataset indices. */
    chosen_on?: number[] | null;
    reported_on?: number[] | null;
  } | null;
  /** Sets of sites: each set against the set that replaces the whole universe. */
  circuit?: CircuitInfo | null;
  /** Attribution patching: one gradient, or integrated gradients over steps. */
  attribution?: { method: AttributionMethod; steps?: number | null } | null;
}

export interface SummaryMetric {
  kind: MetricKind | string;
  target?: Side | null;
  /** The metric in words, such as "logit difference" (absent in older runs). */
  label?: string | null;
  normalization: Normalization;
  denominator: number | null;
  description: string;
  normalized_effect: string;
}

export interface SummaryStatistics {
  bootstrap: number;
  ci: number;
  seed: number;
  method: string;
  /** The meta field whose values group prompts into clusters, and how many clusters there were. */
  cluster?: string | null;
  clusters?: number | null;
  /** The two-sided tail each site's simultaneous band keeps, and what the bands and q-values mean. */
  band_level?: number | null;
  multiple_comparisons?: string | null;
}

export interface PromptSetStats {
  /** log P(answer) − log P(distractor): the logit difference for single tokens. */
  logit_diff: GroupStats;
  answer_prob: GroupStats;
  prefers_answer: number;
  /** The spec's metric (absent in older runs, which measured the logit difference). */
  metric?: GroupStats | null;
}

/** A steered site and strength against its random control, from the same resamples: the
 * difference of their effects' magnitudes, |direction| − |control|. */
export interface ControlComparison {
  row: number;
  coefficient: number;
  index: number;
  control_index: number;
  difference: number | null;
  lo: number | null;
  hi: number | null;
  beats_control: boolean;
}

export interface CircuitRow {
  index: number;
  label: string;
  complement: boolean;
  size: number;
  /** The set's effect as a share of the set that replaces everything. */
  share?: Stat | null;
  /** For a set that keeps its sites: 1 − share, the part of the behavior they carry alone. */
  faithfulness?: Stat | null;
  /** For a keeping set one site short of another: how much faithfulness that site adds. */
  without?: { of: string; site: string; drop: Stat } | null;
  /** For a set of two sites also run alone: effect(both) − effect(a) − effect(b). */
  interaction?: { a: string; b: string; effect: Stat } | null;
}

export interface CircuitInfo {
  universe: UniverseKind[] | null;
  /** The index of the set that replaces the whole universe, if the scope has one. */
  everything: number | null;
  rows: CircuitRow[];
}

export interface DirectSplit {
  prompts: "clean" | "corrupt";
  logit_diff: number | null;
  embeddings: number | null;
  attention: number | null;
  mlp: number | null;
  biases: number | null;
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
  kind: "robustness" | "rerun" | "verification";
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
  /** Per layer, the largest normalized effect (signed): writes the run's logogram. */
  profile?: number[] | null;
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
  /** The spec's metric: patched, in the receiver prompt and in the reference prompt. */
  patched_metric?: number | null;
  receiver_metric?: number | null;
  reference_metric?: number | null;
  /** log P(answer) − log P(distractor): the logit difference for single tokens. */
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

/** An answer or distractor: one string (a token, or a continuation of several tokens), or a set of
 * single tokens any of which counts. */
export type Answer = string | string[];

export interface PromptRecord {
  clean: string;
  corrupt: string;
  answer: Answer;
  distractor: Answer;
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
  /** With a model loaded: the prompts whose answer or distractor is a continuation of several of
   * its tokens, which only the metrics that read continuations can score. */
  continuations?: number[];
}

/** An answer or distractor as the loaded model reads it: its tokens, or the members of a set. */
export interface AnswerTokens {
  text: string;
  tokens: string[];
  id: number | null;
  /** A set of single tokens, any of which counts (its tokens are the members). */
  alternatives?: boolean;
}

export interface TokenStripData {
  clean: { tokens: string[]; ids: number[] };
  corrupt: { tokens: string[]; ids: number[] };
  aligned: boolean;
  differs: number[];
  labels: Record<string, number>;
  answer: AnswerTokens;
  distractor: AnswerTokens;
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
  /** log P(answer) − log P(distractor): the logit difference for single tokens. */
  clean_logit_diff: number | null;
  corrupt_logit_diff: number | null;
  clean_answer_prob: number | null;
  corrupt_answer_prob: number | null;
  /** The requested metric (absent from servers that measured the logit difference only). */
  clean_metric?: number | null;
  corrupt_metric?: number | null;
  clean_top: TopToken[];
  corrupt_top: TopToken[];
}

/** The requested metric over the prompts, beside the preference the baseline always reports. */
export interface BaselineMetric {
  kind: MetricKind | string;
  target: Side | null;
  label: string;
  description: string;
  clean: number | null;
  corrupt: number | null;
  /** clean − corrupt. */
  gap: number | null;
}

export interface BaselineReport {
  n: number;
  prompts: BaselinePrompt[];
  issues: PromptIssue[];
  summary: {
    /** Means of log P(answer) − log P(distractor), the logit difference for single tokens. */
    clean_logit_diff: number | null;
    corrupt_logit_diff: number | null;
    gap: number | null;
    clean_prefers_answer: number;
    corrupt_prefers_answer: number;
    clean_answer_prob: number | null;
    corrupt_answer_prob: number | null;
    metric?: BaselineMetric | null;
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
  kind: ResultKind;
  layer: number;
  head: number | null;
  feature?: number | null;
  position_key: string;
  variant_key?: string | null;
  index_a: number;
  index_b: number;
  row: number;
  col: number;
  effect_a: Stat;
  effect_b: Stat;
  rank_a: number;
  rank_b: number;
  /** "differs": the paired, prompt-by-prompt difference's interval excludes zero. */
  flags: ("sign" | "left_top" | "entered_top" | "differs")[];
  /** b − a in per-prompt effects, when both runs measured the same prompts. */
  difference?: { mean: number; lo: number; hi: number } | null;
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
  /** Whether both runs measured the same prompts, so effects were compared prompt by prompt. */
  paired?: boolean;
  /** Sites whose paired difference excludes zero. */
  n_differs?: number;
}

export interface MemoryEstimate {
  device: string;
  dtype: Dtype;
  n_params: number;
  weights: number;
  activations: number;
  /** Extra memory for a moment while the model loads, when its weights are processed. */
  processing: number;
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
  /** Whether TransformerLens can load this architecture; if not, why (known before downloading). */
  supported: boolean;
  support_note: string | null;
  estimate: MemoryEstimate;
}

export interface IOITemplate {
  id: string;
  text: string;
  default: boolean;
}

/** A setting of a task: true or false, one of `allowed`, or a list of distinct `allowed` values. */
export interface TaskOption {
  name: string;
  type: "bool" | "choice" | "choices";
  default: boolean | string | string[];
  description: string;
  allowed: string[];
}

/** A task Logogram generates prompts for (src/logogram/tasks.py). */
export interface TaskInfo {
  id: string;
  name: string;
  description: string;
  /** The metric that reads its answers, and the option values that change it. */
  metric: MetricKind;
  metric_when: { option: string; value: unknown; metric: MetricKind }[];
  options: TaskOption[];
  templates: IOITemplate[];
}

export interface DatasetCreated {
  name: string;
  path: string;
  n: number;
}

export interface TaskDatasetCreated extends DatasetCreated {
  /** The metric that reads the new dataset's answers. */
  metric: MetricKind;
}
