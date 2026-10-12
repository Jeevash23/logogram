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
  device: "cpu" | "cuda" | "mps" | `cuda:${number}`;
  device_name: string;
  process_weights: boolean;
  /** Where the weights came from (Hugging Face when absent), and a local model's files' hash. */
  source?: "hub" | "local";
  sha256?: string | null;
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
    /** Whether it is made for chat (prompts run as plain text, without its chat template). */
    chat?: ChatFormat;
    /** The SHA-256 of the chat template its tokenizer uses by default: prompts can be wrapped in
     * it (a spec's tokenization.chat_template). Null when it has none. */
    chat_template_sha256?: string | null;
    /** A soft cap on its logits, or null. */
    logit_soft_cap?: number | null;
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
  /** checking: a local model's files are checked against their hash before it loads. */
  stage?: "resolving" | "downloading" | "checking" | "loading" | "processing" | "ready";
  done?: number;
  total?: number;
  file?: string;
  error?: string;
  revision?: string;
  info?: ModelInfo;
  memory?: number | null;
  ref?: ModelRef;
}

export interface Job {
  project_session?: string | null;
  id: string;
  kind: "load_model" | "load_sae" | "run" | "add_model";
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

/** A run waiting for the job in progress to end, in the open project's queue. */
export interface QueuedRun {
  id: string;
  /** The spec's name. */
  title: string;
  spec: Spec;
  draft_id: string | null;
  derived_from: DerivedFrom | null;
  project_session: string;
  /** Seconds since the epoch, like a job's start. */
  queued_at: number;
}

/** How the open project's queue stands: "waiting" (its next run starts when the job in progress
 * ends), "running" (one of its runs is in progress, here or in another process, and the next
 * follows), "paused" (nothing starts until it is started: a queue that comes back with its project
 * is paused) or "idle" (empty). */
export type QueueState = "idle" | "waiting" | "running" | "paused";

/** The open project's queue, kept in the project, and how it stands (GET /api/queue/state and
 * the "queue" event). */
export interface QueueStatus {
  project_session: string | null;
  runs: QueuedRun[];
  state: QueueState;
  /** Who runs it, when another process does: "`logogram queue run` (process 1234)". */
  runner?: string | null;
  /** Why the project's saved queue can't be read, and how to fix it. */
  error?: string | null;
}

/** A run sent to the queue: started at once when nothing was running (run_id and job), or
 * waiting its turn (queued). */
export interface QueueAdded {
  run_id: string | null;
  job: Job | null;
  queued: QueuedRun | null;
  queue: QueuedRun[];
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
  /** Whether single keys act as shortcuts (on until turned off). */
  single_key_shortcuts?: boolean;
  projects_parent: string;
  update: UpdateStatus;
  sae?: SAEStatus;
  /** The runs waiting for the job in progress, first in line first. */
  queue?: QueuedRun[];
  /** The same runs, how the queue stands, and who runs it. */
  queue_status?: QueueStatus | null;
}

export interface RecentProject {
  path: string;
  name: string;
  opened: string | null;
  modified: string;
}

/** A bundled example (logogram.gallery): what it asks, the result to expect, and its spec. */
export interface ExampleInfo {
  id: string;
  name: string;
  question: string;
  expect: string;
  /** The publication the result is read against. */
  source: string;
  /** Where its copy goes in the projects folder, and whether one is there already (opening the
   * example then opens that copy as it was left). */
  path: string;
  copied: boolean;
  experiment: string;
  n_prompts: number;
  spec: Spec;
}

export interface FolderListing {
  path: string;
  parent: string | null;
  is_project: boolean;
  entries: { name: string; path: string; is_project: boolean }[];
}

/** A problem report to paste into an issue (logogram.support): its rows, and the same as text. */
export interface ProblemReport {
  rows: { label: string; value: string }[];
  /** Whether the processor, GPUs, memory and operating system's version are in it. */
  hardware: boolean;
  text: string;
  new_issue_url: string;
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
  /** Each GPU, by its index among those PyTorch sees (a spec names one as cuda:N); Apple's has no
   * index. */
  gpus: { name: string; memory_total: number; memory_free: number | null; bf16: boolean; index?: number | null }[];
  /** The GPU "cuda" (and "auto") runs on. */
  default_gpu?: number | null;
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
/** Where the model runs: `cuda:N` is one GPU of several. */
export type Device = "auto" | "cpu" | "cuda" | "mps" | `cuda:${number}`;
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
  /** The site it writes: heads' outputs (z, every head of the layer side by side) for "head". */
  site: StreamKind | "head";
  layer: number;
  d_in: number;
  d_sae: number;
  activation: string;
  k: number | null;
  normalize: string;
  format: string;
  note: string;
  fit: SAEFit | null;
  /** The site it reads: the MLP's input ("mlp_in") for a transcoder. */
  site_in?: string;
  transcoder?: boolean;
  /** A transcoder that also adds its input times a skip matrix to what it writes. */
  skip?: boolean;
}

/** What one feature writes directly into the logits: one unit of its decoder direction, read
 * through the final normalization (its scale held at its value at one token) and the
 * unembedding, skipping every later layer. Logit changes are relative to the vocabulary's mean. */
export interface FeatureWrites {
  feature: number;
  site: StreamKind | "head";
  layer: number;
  index: number;
  which: "clean" | "corrupt";
  tokens: string[];
  /** The token it was read at, and why that one: chosen, where the feature fires most in the
   * prompt, or the last token (it fires nowhere in the prompt). */
  position: number;
  token: string;
  chosen: "given" | "fires_most" | "last";
  /** The feature's activation at that token. */
  activation: number;
  promoted: { id: number; token: string; logit: number }[];
  suppressed: { id: number; token: string; logit: number }[];
  /** The length of one unit of the feature in the residual stream. */
  norm: number;
  /** The model's soft cap on its logits (the values are before it), or null. */
  soft_cap: number | null;
}

/** Whether a loaded model is made for chat: Logogram runs prompts as plain text, without the chat
 * template such models are trained on. */
export interface ChatFormat {
  instruct: boolean;
  /** Whether its tokenizer has a chat template. */
  template: boolean;
  /** Why it counts as made for chat, or null. */
  reason: string | null;
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

/** What steering adds: the mean difference of the training pairs, or one SAE feature's decoder
 * direction where the SAE reads. */
export type SteeringDirection = { kind: "mean_difference" } | { kind: "sae_feature"; feature: number };

/** Greedy continuations of the first held-out prompts, unsteered and at each strength. */
export interface Continuations {
  prompts: number;
  tokens: number;
}

export type ExperimentSpec =
  | { kind: "activation_patching"; direction: Direction }
  | { kind: "ablation"; baseline: BaselineSpec }
  | { kind: "direct_logit_attribution"; prompts: "clean" | "corrupt" }
  | { kind: "attribution_patching"; direction: Direction; method: AttributionMethod; steps: number | null }
  | { kind: "path_patching"; direction: Direction; receivers: PathReceiverSpec[]; freeze_mlps: boolean }
  | {
      kind: "steering";
      apply_to: "clean" | "corrupt";
      direction: SteeringDirection;
      coefficients: number[];
      train_fraction: number;
      seed: number;
      control: boolean;
      continuations: Continuations | null;
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

/** The model of a spec: from Hugging Face, pinned by its revision, or a local folder found by the
 * hash of its files (`sha256`). */
export interface ModelRef {
  id: string;
  source: "hub" | "local";
  revision: string | null;
  sha256: string | null;
  dtype: Dtype;
  device: Device;
  process_weights: boolean;
}

/** How prompts are wrapped in the model's own chat template: as the person's message, or as the
 * start of the model's reply to a fixed message. */
export type ChatTemplateSpec =
  | { kind: "user_turn"; system: string | null }
  | { kind: "assistant_prefix"; system: string | null; user: string };

/** A version 3 spec: every choice that can change a number is stated. */
export interface Spec {
  logogram_spec: 3;
  name: string;
  notes: string;
  model: ModelRef;
  dataset: { path: string; sha256: string | null; limit: number | null };
  tokenization: { prepend_bos: boolean; chat_template: ChatTemplateSpec | null };
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
   * each row's vector at strength 1, each steered site and strength against its random control,
   * and for an SAE feature what strength 1 is. */
  steering?: {
    train: number[];
    test: number[];
    norms: number[];
    control?: ControlComparison[] | null;
    feature?: SteeringFeature | null;
  } | null;
  /** Runs on SAE features: the SAE, its fit on these prompts, and (attribution patching) how much
   * of the site's estimated effect its features account for. */
  features?: {
    sae: SAEInfo;
    fit: SAEFit;
    site_estimate?: number | null;
    features_estimate?: number | null;
    /** Through a transcoder's skip connection, which no feature carries. */
    skip_estimate?: number | null;
    evaluated?: number | null;
    /** Every feature, chosen on some prompts and reported on the others: their dataset indices. */
    chosen_on?: number[] | null;
    reported_on?: number[] | null;
  } | null;
  /** Sets of sites: each set against the set that replaces the whole universe. */
  circuit?: CircuitInfo | null;
  /** Attribution patching: one gradient, or integrated gradients over steps. */
  attribution?: { method: AttributionMethod; steps?: number | null } | null;
  /** How the prompts were wrapped in the model's own chat template: the template's SHA-256, and
   * its text before and after each prompt. */
  chat_template?: {
    kind: ChatTemplateSpec["kind"];
    system: string | null;
    user: string | null;
    sha256: string;
    before: string;
    after: string;
  } | null;
  /** A run that stopped before the end: what it kept. Absent or null for a finished run. */
  partial?: PartialRun | null;
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
/** Steering along an SAE feature: the feature and its SAE, where the SAE writes (the site its
 * decoder direction is added to; "head" is every head's output z), and for each row of the
 * layout what strength 1 is: the feature's typical activation, the mean of its positive
 * activations at the row's token of the training pairs' clean and corrupt prompts (`fired` of
 * the `measured` activations were positive). */
export interface SteeringFeature {
  feature: number;
  sae: SAERef & { revision: string };
  site: StreamKind | "head";
  layer: number;
  transcoder: boolean;
  /** The SAE standardizes its inputs: each prompt's vector is also scaled by its input's spread. */
  standardized: boolean;
  activations: number[];
  fired: number[];
  measured: number[];
}

/** Tokens a model generated, and each as its tokenizer writes it. */
export interface GeneratedTokens {
  ids: number[];
  tokens: string[];
}

/** A steering run's continuations.json: greedy continuations of the first held-out prompts,
 * without steering and with each row's vector at each strength (and its random control). */
export interface ContinuationsFile {
  logogram_continuations: 1;
  run_id: string;
  receiver: Side;
  tokens: number;
  decoding: "greedy";
  rows: { row: number; label: string }[];
  prompts: {
    index: number;
    /** The steered prompt as written, and as the model read it. */
    text: string;
    prompt: string[];
    unsteered: GeneratedTokens;
    steered: (GeneratedTokens & { row: number; coefficient: number; control: boolean; key: string })[];
  }[];
}

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
    /** A local model, by the hash of its files. */
    source?: "hub" | "local" | null;
    sha256?: string | null;
    architecture: string;
    process_weights: boolean;
    backend: string;
    backend_version: string;
    /** Whether the model is made for chat: its prompts ran as plain text, without its template. */
    chat?: ChatFormat | null;
  };
  device?: {
    type: string;
    /** Which of the machine's GPUs (cuda:N). */
    index?: number | null;
    name: string;
    cuda: string | null;
    threads: number;
    deterministic_algorithms: boolean;
  };
  dtype?: string;
  /** The prompts the run read: its saved copy, and the dataset it was copied from. */
  dataset?: { path: string; sha256: string; n: number; source_path?: string | null };
  derived_from?: DerivedFrom | null;
  /** The chat template the prompts were wrapped in. */
  chat_template?: { kind: ChatTemplateSpec["kind"]; sha256: string } | null;
}

/** Where a run came from: another run (a robustness check, a rerun, a verification), or a sweep. */
export type DerivedFrom =
  | { run: string; kind: "robustness" | "rerun" | "verification"; change?: string | null }
  | SweepMember;

/** A value a sweep runs: a seed, or a dataset's path or a revision as given. */
export type SweepValue = number | string;

/** A run of a sweep (src/logogram/schema.py SweepMember): every run of it records the sweep. */
export interface SweepMember {
  kind: "sweep";
  sweep: string;
  /** The field it varies, dotted as in the spec, and this run's value and place among the values. */
  field: string;
  value: SweepValue;
  index: number;
  of: number;
  values: SweepValue[];
  /** The run the sweep was made from, and the value it stands in for (not run again). */
  run: string | null;
  original: number | null;
  /** This run's value in words, such as "bootstrap seed 2". */
  change: string | null;
}

export type SweepKind = "seed" | "dataset" | "revision";

/** A field a run's spec can be swept over (GET /api/runs/{id}/sweep-fields). */
export interface SweepField {
  field: string;
  kind: SweepKind;
  label: string;
  plural: string;
  description: string;
  current: SweepValue | null;
  current_label: string | null;
  /** What the app offers at first: the spec's own value first, so its run stands in for it. */
  suggested: SweepValue[];
}

/** What a sweep is asked for: a run (or a spec) and the values of one of its fields, in order. */
export interface SweepRequest {
  run?: string;
  spec?: Spec;
  field: string;
  values: SweepValue[];
}

/** The runs a sweep makes, one per value (POST /api/sweeps/plan). */
export interface SweepPlan {
  sweep: string;
  field: string;
  kind: SweepKind;
  label: string;
  plural: string;
  values: SweepValue[];
  labels: string[];
  original: string | null;
  stands_in: number | null;
  runs: { index: number; value: SweepValue; label: string; name: string | null; original: boolean }[];
}

export interface SweepStarted {
  sweep: string;
  plan: SweepPlan;
  run_id: string | null;
  job: Job | null;
  queue: QueuedRun[];
}

export type SweepStatus = "finished" | "failed" | "cancelled" | "running" | "draft" | "missing" | "deleted";

export interface SweepValueRow {
  index: number;
  value: SweepValue;
  label: string;
  run: string | null;
  status: SweepStatus;
  /** The run the sweep was made from, standing in for this value. */
  original: boolean;
  title: string | null;
  error: string | null;
  n_prompts: number | null;
  /** The commit a revision resolved to, and the prompts a dataset held, as the run pinned them. */
  revision: string | null;
  dataset_sha256: string | null;
}

/** One site at one value: the run's mean effect, its interval and SD over prompts, and n. */
export interface SweepPoint {
  mean: number | null;
  lo: number | null;
  hi: number | null;
  sd: number | null;
  n: number | null;
}

export type SweepVerdict = "holds" | "size" | "sign" | "unknown";

export interface SweepSite {
  key: string;
  rank: number;
  label: string;
  kind: ResultKind;
  layer: number;
  head: number | null;
  feature: number | null;
  position: PositionSpec | null;
  position_key: string;
  variant: SiteBase["variant"];
  variant_key: string | null;
  /** One per value; null where the value has no results for this site (never zero). */
  points: (SweepPoint | null)[];
  measured: number;
  /** The spread of its means across the values that measured it. */
  spread: { mean: number; sd: number | null; min: number; max: number } | null;
  /** The median half-width of the runs' own intervals. */
  half_width: number | null;
  above: number[];
  below: number[];
  /** Two values whose intervals exclude zero on opposite sides; two whose intervals exclude each other. */
  sign: number[] | null;
  apart: number[] | null;
  missing: number[];
  excludes_zero: "every" | "some" | "none";
  verdict: SweepVerdict;
  text: string;
}

/** A sweep read back (GET /api/sweeps/{id}). */
export interface SweepTable {
  sweep: string;
  field: string;
  kind: SweepKind;
  label: string;
  plural: string;
  description: string;
  name: string;
  created: string | null;
  original: string | null;
  stands_in: number | null;
  values: SweepValueRow[];
  finished: number;
  choose: "first" | "all";
  top: number;
  chosen: string;
  problem: string | null;
  n_sites: number;
  sites: SweepSite[];
  counts: Record<SweepVerdict, number>;
  headline: string;
  spread: string;
  notes: string[];
  ci: number | null;
  measure: Measure | null;
  metric: SummaryMetric | null;
  experiment: ExperimentSpec | null;
}

/** A model's branches and tags on Hugging Face, each with its commit (GET /api/models/revisions). */
export interface Revisions {
  id: string;
  revisions: { name: string; commit: string; kind: "branch" | "tag" }[];
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
  /** Running in another process (logogram run, a notebook, another server): its progress isn't
   * streamed here, so the app reads the history again to see it end. */
  elsewhere?: boolean;
  /** The dataset the run read, as it was named when it ran (`dataset` is then its snapshot). */
  dataset_source?: string | null;
  /** What a person says about the run, kept beside its spec (annotations.json): a title shown
   * instead of the spec's name, a star, archived (kept out of the history until shown). */
  title?: string | null;
  starred?: boolean;
  archived?: boolean;
  /** Why the run's annotations.json was ignored, when it can't be read. */
  annotations_error?: string | null;
  /** The run's spec can't be read: `error` says what is wrong and how to fix it. Its folder can
   * be deleted; it can't be opened. */
  damaged?: boolean;
  /** A cancelled or failed run that kept what finished. */
  partial?: PartialRun | null;
}

/** What a run that stopped before the end (cancelled, or failed mid-sweep) kept. A sweep that goes
 * layer by layer (site by site, set by set) keeps the sites every prompt ran at; a method that
 * measures every site at once keeps every site over the prompts that finished. Statistics come
 * from the kept rows as for a finished run. */
export interface PartialRun {
  status: "cancelled" | "failed";
  unit: "layers" | "sites" | "sets" | "prompts";
  done: number;
  total: number;
  sites: number;
  sites_total: number;
  prompts: number;
  prompts_total: number;
  /** "Stopped after 7 of 12 layers: these are the layers that finished." */
  note: string;
  /** What the run doesn't keep, or keeps over the finished part only. */
  left_out: string[];
}

/** A change to what is said about a run. What is left out stays; a title of "" or null (or the
 * spec's own name) shows the spec's name again. */
export interface RunAnnotationsChange {
  title?: string | null;
  starred?: boolean;
  archived?: boolean;
}

export interface RunDetail {
  id: string;
  spec: Spec;
  summary: Summary | null;
  manifest: Manifest | null;
  predictions?: PredictionReport | null;
  /** What a steering run's model went on to say, when the run recorded it. */
  continuations?: ContinuationsFile | null;
  listing: RunListing | null;
  folder: string;
  /** A failed run's traceback, without the paths and names of the machine it ran on. */
  traceback?: string | null;
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
  /** With a chat template: the clean tokens that hold the prompt (first, one past the last); the
   * others are the template's. */
  prompt?: [number, number] | null;
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

export interface LookToken {
  id: number;
  token: string;
  prob: number;
}

/** Any text read without a dataset (POST /api/look). */
export interface LookReport {
  tokens: string[];
  ids: number[];
  prepend_bos: boolean;
  /** The chat template the text was wrapped in, and the tokens that hold the text itself. */
  chat_template?: ChatTemplateSpec | null;
  text_tokens?: [number, number] | null;
  top_k: number;
  position: number;
  /** After each token: the model's most likely next tokens. */
  predictions: LookToken[][];
  /** At `position`: what each layer would predict, when the logit lens holds for this model. */
  lens: { layer: number; top: LookToken[] }[] | null;
  lens_note: string | null;
}

export interface LookAttention {
  tokens: string[];
  layer: number;
  head: number;
  pattern: number[][];
}

export interface LookFeatures {
  tokens: string[];
  features: { feature: number; activation: number }[][];
  active: number[];
  first_real_token: number;
}

/** One SAE feature along any text: its activation at each token, and its largest from the first
 * real token on (POST /api/look/feature). */
export interface LookFeatureAlong {
  tokens: string[];
  feature: number;
  activations: number[];
  largest: number;
  position: number;
  first_real_token: number;
}

/** What to add while a text continues: `activation` units of a feature of the loaded SAE, at
 * every position from token `start` on. */
export interface LookSteer {
  feature: number;
  activation: number;
  start: number;
}

/** What the model writes after a text, greedily, and with a feature added if asked
 * (POST /api/look/continue). */
export interface LookContinuation {
  tokens: string[];
  ids: number[];
  prepend_bos: boolean;
  new: number;
  plain: GeneratedTokens;
  steer: (LookSteer & { site: string; layer: number; where: string; largest: number; largest_at: number }) | null;
  steered: GeneratedTokens | null;
}

export type HeadKindName = "previous_token" | "duplicate_token" | "induction";

export interface HeadKindScores {
  description: string;
  /** [layer][head]: the mean over sequences and its bootstrap interval. */
  mean: number[][];
  lo: number[][];
  hi: number[][];
  /** [sequence][layer][head]: the values the statistics come from. */
  per_sequence: number[][][];
}

export interface TopHead {
  layer: number;
  head: number;
  mean: number;
  lo: number;
  hi: number;
}

/** Every head scored by what it attends to on random tokens repeated twice (POST /api/heads/kinds). */
export interface HeadKindsReport {
  settings: { seed: number; sequences: number; length: number; prepend_bos: boolean; batch_size: number; bootstrap: number; ci: number; chat_template?: ChatTemplateSpec | null };
  n_layers: number;
  n_heads: number;
  kinds: Record<HeadKindName, HeadKindScores>;
  top: Record<HeadKindName, TopHead[]>;
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
  /** Versions or devices the two manifests differ in: these alone can change the last digits. */
  environment_differences?: EnvironmentDifference[];
  /** A run that stopped early: what it kept (only the sites both runs have are compared). */
  partial_a?: PartialRun | null;
  partial_b?: PartialRun | null;
}

export interface EnvironmentDifference {
  /** A version ("torch"), or a device's "device.type", "device.name" or "device.cuda". */
  key: string;
  label: string;
  a: string;
  b: string;
}

/** What the server's results depend on beyond a spec (GET /api/environment). */
export interface Environment {
  logogram: string;
  python: string;
  implementation: string;
  os: string;
  architecture: string;
  /** Named as in run manifests: torch, transformer_lens, transformers, numpy, … */
  versions: Record<string, string | null>;
  torch_cuda: string | null;
  devices: string[];
  /** What a spec's "auto" device runs on here. */
  default_device: string;
}

/** What a run would need beyond the loaded model's weights (POST /api/models/run-memory). */
export interface RunMemory {
  device: string;
  batch_size: number;
  seq_len: number;
  /** Whether the method holds every layer's activations at once. */
  all_layers: boolean;
  activations: number;
  total: number;
  available: number;
  verdict: "fits" | "tight" | "wont_fit";
  explanation: string;
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
  /** Null for a local model, which has no revisions. */
  revision: string | null;
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

/** A dataset's prompt pairs, some edited and some left out, to save as a new dataset (POST
 * /api/datasets/derive). The dataset they come from never changes. */
export interface DeriveRequest {
  /** The dataset edited, by its path in the project, and the version of it that was edited. */
  dataset: string;
  dataset_sha256: string;
  name: string;
  /** Indices of the prompt pairs to leave out, and the new value of each edited one by index. */
  remove: number[];
  replace: Record<number, PromptRecord>;
}

export interface TaskDatasetCreated extends DatasetCreated {
  /** The metric that reads the new dataset's answers. */
  metric: MetricKind;
}

/** A prompt in the loaded model's own chat template: the whole text, where the prompt is in it
 * (null when the template changes the prompt, as `problem` says), the template's text before and
 * after a prompt, and its SHA-256. */
export interface ChatWrapped {
  text: string;
  start: number | null;
  end: number | null;
  before: string;
  after: string;
  sha256: string;
  problem: string | null;
}

/** A model added from a folder on this machine. Specs name it by its name and the hash of its
 * files; the folder is shown on this machine only. */
export interface LocalModel {
  name: string;
  folder: string;
  sha256: string;
  added: string;
  size: number;
  files: number;
}
