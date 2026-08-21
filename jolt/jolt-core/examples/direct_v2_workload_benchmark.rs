//! D20/D21 real-workload benchmark for direct block-Jolt V2.
//!
//! This runner never creates a native `JoltProof`. A shape-specific setup is
//! prepared once from a bounded trace audit, then a fresh lazy trace is proved
//! through compact Jolt relations, Nova folding, deferred Dory closure, and
//! one Spartan-compressed final artifact.

extern crate jolt_inlines_keccak256;

use std::{
    error::Error,
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_core::{
    host::Program,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        block::{
            audit_block_jolt_ram_endpoints, audit_block_jolt_trace,
            compress_block_jolt_final_proof_with_metrics, prepare_block_jolt_nova_setup,
            prove_block_jolt_streaming_with_setup, BlockJoltFinalizationMetrics,
            BlockJoltSetupMetrics, BlockJoltStreamingMetrics, DirectChunkedPreprocessing,
            BLOCK_JOLT_PROTOCOL_VERSION,
        },
        program::ProgramPreprocessing,
        verifier::JoltSharedPreprocessing,
    },
};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

const D20_SCHEMA_VERSION: u16 = 1;
const D20_DIGEST_DOMAIN: &[u8] = b"direct-block-jolt-d20-artifact-v1";
const DEFAULT_OUTPUT: &str = "benchmark-runs/direct-v2-d20/fibonacci-2-cap128.json";

#[derive(Clone, Debug, Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Workload::Fibonacci)]
    workload: Workload,

    /// Fibonacci n, SHA3 iteration count, or Collatz input.
    #[arg(long, default_value_t = 2)]
    scale: u32,

    #[arg(long, default_value_t = 128)]
    block_capacity: usize,

    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    #[arg(long)]
    markdown_output: Option<PathBuf>,

    #[arg(long, default_value = "target-direct-v2-d20-guest")]
    guest_target: PathBuf,

    /// Cargo profile used to build this executable; recorded in the artifact.
    #[arg(long, default_value = "release")]
    build_profile_label: String,

    #[arg(long, default_value_t = 10)]
    memory_sample_interval_ms: u64,

    /// Build and audit the real trace, derive RAM K, then stop before setup.
    #[arg(long, default_value_t = false)]
    trace_audit_only: bool,

    /// Audit all real-trace RAM endpoints without Nova/Dory setup.
    #[arg(long, default_value_t = false)]
    ram_audit_only: bool,

    /// Run the real bounded trace audit and reusable Nova/Spartan setup, then
    /// stop before per-block proving. Useful for selecting a feasible shape.
    #[arg(long, default_value_t = false)]
    setup_only: bool,

    /// Optional D20 artifact used for an exact-workload D21 comparison.
    #[arg(long)]
    baseline_input: Option<PathBuf>,

    #[arg(long, default_value_t = 15.0)]
    max_regression_percent: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Workload {
    Fibonacci,
    Sha3Chain,
    MemoryOps,
    Collatz,
}

impl Workload {
    fn guest_name(self) -> &'static str {
        match self {
            Self::Fibonacci => "fibonacci-guest",
            Self::Sha3Chain => "sha3-chain-guest",
            Self::MemoryOps => "memory-ops-guest",
            Self::Collatz => "collatz-guest",
        }
    }

    fn configure_program(self, program: &mut Program) {
        if self == Self::Collatz {
            program.set_func("collatz_convergence");
        }
        if self == Self::MemoryOps {
            program.set_heap_size(1 << 16);
        }
    }

    fn memory_config(self, program_size: u64) -> MemoryConfig {
        MemoryConfig {
            program_size: Some(program_size),
            heap_size: if self == Self::MemoryOps {
                1 << 16
            } else {
                MemoryConfig::default().heap_size
            },
            ..MemoryConfig::default()
        }
    }

    fn max_trace_length(self) -> usize {
        match self {
            Self::Fibonacci | Self::MemoryOps => 1 << 16,
            Self::Collatz => 1 << 20,
            Self::Sha3Chain => 1 << 22,
        }
    }

    fn input_bytes(self, scale: u32) -> Result<Vec<u8>, String> {
        let mut inputs = Vec::new();
        match self {
            Self::Fibonacci => {
                inputs.extend(postcard::to_stdvec(&scale).map_err(|error| error.to_string())?)
            }
            Self::Sha3Chain => {
                inputs.extend(postcard::to_stdvec(&[5u8; 32]).map_err(|error| error.to_string())?);
                inputs.extend(postcard::to_stdvec(&scale).map_err(|error| error.to_string())?);
            }
            Self::MemoryOps => {}
            Self::Collatz => inputs.extend(
                postcard::to_stdvec(&(scale as u128).max(2)).map_err(|error| error.to_string())?,
            ),
        }
        Ok(inputs)
    }

    fn id(self, scale: u32) -> String {
        match self {
            Self::Fibonacci => format!("fibonacci-{scale}"),
            Self::Sha3Chain => format!("sha3-chain-{scale}-iterations"),
            Self::MemoryOps => "memory-ops".to_string(),
            Self::Collatz => format!("collatz-{scale}"),
        }
    }
}

struct PreparedWorkload {
    program: Program,
    elf: Vec<u8>,
    inputs: Vec<u8>,
    memory_config: MemoryConfig,
    preprocessing: DirectChunkedPreprocessing,
    workload: String,
    workload_digest: [u8; 32],
    preprocessing_micros: u128,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PhaseMemory {
    start_bytes: Option<u64>,
    peak_bytes: Option<u64>,
    peak_delta_bytes: Option<u64>,
}

struct PeakMemorySampler {
    start: Option<u64>,
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakMemorySampler {
    fn start(interval_ms: u64) -> Self {
        let start = physical_memory_bytes();
        let peak = Arc::new(AtomicU64::new(start.unwrap_or_default()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_peak = Arc::clone(&peak);
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                if let Some(memory) = physical_memory_bytes() {
                    thread_peak.fetch_max(memory, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
            if let Some(memory) = physical_memory_bytes() {
                thread_peak.fetch_max(memory, Ordering::Relaxed);
            }
        });
        Self {
            start,
            peak,
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> PhaseMemory {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let peak = self
            .start
            .map(|start| self.peak.load(Ordering::Relaxed).max(start));
        PhaseMemory {
            start_bytes: self.start,
            peak_bytes: peak,
            peak_delta_bytes: self
                .start
                .zip(peak)
                .map(|(start, peak)| peak.saturating_sub(start)),
        }
    }
}

impl Drop for PeakMemorySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D20Platform {
    os: String,
    arch: String,
    build_profile: String,
    rustc: String,
    git_commit: String,
    rayon_threads: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D20BenchmarkSample {
    verified: bool,
    setup_total_micros: u128,
    prove_stream_total_micros: u128,
    finalization_total_micros: u128,
    final_decode_and_verify_micros: u128,
    cycles_per_second: f64,
    setup_metrics: BlockJoltSetupMetrics,
    streaming_metrics: BlockJoltStreamingMetrics,
    finalization_metrics: BlockJoltFinalizationMetrics,
    setup_memory: PhaseMemory,
    prove_stream_memory: PhaseMemory,
    finalization_memory: PhaseMemory,
    final_verify_memory: PhaseMemory,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D20Comparison {
    baseline_digest: String,
    prove_stream_percent: f64,
    pcs_spool_percent: f64,
    final_artifact_percent: f64,
    peak_memory_percent: Option<f64>,
    passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D20BenchmarkArtifact {
    schema_version: u16,
    protocol_version: String,
    platform: D20Platform,
    workload: String,
    workload_sha3_256: String,
    block_capacity: usize,
    derived_ram_k: usize,
    setup_id: String,
    preprocessing_micros: u128,
    sample: D20BenchmarkSample,
    comparison: Option<D20Comparison>,
    artifact_digest: String,
}

impl D20BenchmarkArtifact {
    fn seal(&mut self) -> Result<(), serde_json::Error> {
        self.artifact_digest.clear();
        self.artifact_digest = hex_digest(digest_json(self)?);
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != D20_SCHEMA_VERSION
            || self.protocol_version != BLOCK_JOLT_PROTOCOL_VERSION
            || self.workload.is_empty()
            || self.block_capacity < 2
            || !self.block_capacity.is_power_of_two()
            || self.derived_ram_k == 0
            || !self.derived_ram_k.is_power_of_two()
            || !self.sample.verified
            || self.sample.streaming_metrics.block_count == 0
            || self.sample.streaming_metrics.total_active_cycles == 0
            || !self
                .sample
                .streaming_metrics
                .residency_is_bounded(self.block_capacity)
            || self.sample.finalization_metrics.final_artifact_bytes == 0
        {
            return Err("D20 artifact has an invalid shape or failed invariant".to_string());
        }
        let mut canonical = self.clone();
        canonical.artifact_digest.clear();
        let expected = hex_digest(digest_json(&canonical).map_err(|error| error.to_string())?);
        if self.artifact_digest != expected {
            return Err("D20 artifact digest mismatch".to_string());
        }
        Ok(())
    }

    fn compare(&self, baseline: &Self, limit: f64) -> Result<D20Comparison, String> {
        baseline.validate()?;
        if self.workload != baseline.workload
            || self.workload_sha3_256 != baseline.workload_sha3_256
            || self.block_capacity != baseline.block_capacity
            || self.derived_ram_k != baseline.derived_ram_k
            || self.platform.os != baseline.platform.os
            || self.platform.arch != baseline.platform.arch
        {
            return Err("D20/D21 comparison requires an identical workload and shape".to_string());
        }
        let prove_stream_percent = percent_change(
            self.sample.prove_stream_total_micros as f64,
            baseline.sample.prove_stream_total_micros as f64,
        );
        let pcs_spool_percent = percent_change(
            self.sample.streaming_metrics.spooled_pcs_witness_bytes as f64,
            baseline.sample.streaming_metrics.spooled_pcs_witness_bytes as f64,
        );
        let final_artifact_percent = percent_change(
            self.sample.finalization_metrics.final_artifact_bytes as f64,
            baseline.sample.finalization_metrics.final_artifact_bytes as f64,
        );
        let peak_memory_percent = self
            .sample
            .prove_stream_memory
            .peak_delta_bytes
            .zip(baseline.sample.prove_stream_memory.peak_delta_bytes)
            .map(|(current, old)| percent_change(current as f64, old as f64));
        let passed = prove_stream_percent <= limit
            && final_artifact_percent <= limit
            && peak_memory_percent.is_none_or(|change| change <= limit);
        Ok(D20Comparison {
            baseline_digest: baseline.artifact_digest.clone(),
            prove_stream_percent,
            pcs_spool_percent,
            final_artifact_percent,
            peak_memory_percent,
            passed,
        })
    }

    fn markdown(&self) -> String {
        let m = &self.sample.streaming_metrics;
        let f = &self.sample.finalization_metrics;
        let r = &m.relation_proving;
        let p = &m.pcs_proving;
        format!(
            "# Direct Block-Jolt V2 workload benchmark\n\n\
             - Workload: `{}`\n\
             - Workload digest: `{}`\n\
             - Block capacity / blocks / cycles: {} / {} / {}\n\
             - Derived RAM K: {}\n\
             - Setup ID: `{}`\n\
             - Verified: {}\n\n\
             | Phase | Time (s) |\n|---|---:|\n\
             | Reusable setup | {:.3} |\n\
             | Trace capture | {:.3} |\n\
             | Lookup relation | {:.3} |\n\
             | Register relation | {:.3} |\n\
             | RAM relation | {:.3} |\n\
             | CPU/R1CS relation | {:.3} |\n\
             | PCS polynomial materialization | {:.3} |\n\
             | PCS commitments | {:.3} |\n\
             | Nova folding | {:.3} |\n\
             | PCS spool writes | {:.3} |\n\
             | Dory closure | {:.3} |\n\
             | Spartan compression | {:.3} |\n\
             | Final decode + verify | {:.3} |\n\n\
             | Size | Bytes |\n|---|---:|\n\
             | Trace spool | {} |\n\
             | PCS witness spool (prover-only) | {} |\n\
             | Spartan proof | {} |\n\
             | Dory proof | {} |\n\
             | Final artifact | {} |\n",
            self.workload,
            self.workload_sha3_256,
            self.block_capacity,
            m.block_count,
            m.total_active_cycles,
            self.derived_ram_k,
            self.setup_id,
            self.sample.verified,
            seconds(self.sample.setup_total_micros),
            seconds(m.trace_capture_micros),
            seconds(r.lookup_micros),
            seconds(r.register_micros),
            seconds(r.ram_micros),
            seconds(r.cpu_micros),
            seconds(
                p.lookup_polynomial_micros
                    + p.register_polynomial_micros
                    + p.ram_polynomial_micros
                    + p.cpu_polynomial_micros
            ),
            seconds(p.commitment_micros),
            seconds(m.nova_fold_micros),
            seconds(m.pcs_spool_write_micros),
            seconds(m.deferred_pcs_close_micros),
            seconds(f.spartan_prove_micros),
            seconds(self.sample.final_decode_and_verify_micros),
            m.spooled_trace_bytes,
            m.spooled_pcs_witness_bytes,
            f.compressed_nova_bytes,
            f.deferred_dory_bytes,
            f.final_artifact_bytes,
        )
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("direct_v2_d20_error={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args).map_err(invalid_input)?;
    if !args.trace_audit_only && !args.ram_audit_only && !args.setup_only && cfg!(debug_assertions)
    {
        return Err(invalid_input(
            "cryptographic benchmark must be run with cargo run --release".to_string(),
        )
        .into());
    }
    let prepared = prepare(&args)?;
    if args.trace_audit_only {
        let audit = audit_block_jolt_trace(
            &prepared.preprocessing,
            args.block_capacity,
            trace_blocks(&prepared, args.block_capacity),
        )?;
        println!(
            "direct_v2_d20_trace_audit={}",
            serde_json::to_string(&audit)?
        );
        return Ok(());
    }
    if args.ram_audit_only {
        let audit = audit_block_jolt_ram_endpoints(
            &prepared.preprocessing,
            args.block_capacity,
            trace_blocks(&prepared, args.block_capacity),
        )?;
        println!("direct_v2_d20_ram_audit={}", serde_json::to_string(&audit)?);
        return Ok(());
    }

    eprintln!("direct_v2_d20_phase=setup_start");
    let setup_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let setup_started = Instant::now();
    let (setup, setup_metrics) = prepare_block_jolt_nova_setup(
        &prepared.preprocessing,
        args.block_capacity,
        trace_blocks(&prepared, args.block_capacity),
    )?;
    let setup_total_micros = setup_started.elapsed().as_micros();
    let setup_memory = setup_memory_sampler.finish();
    eprintln!(
        "direct_v2_d20_phase=setup_complete blocks={} cycles={} ram_k={} micros={}",
        setup_metrics.trace.block_count,
        setup_metrics.trace.total_active_cycles,
        setup.config().ram_k,
        setup_total_micros
    );
    if args.setup_only {
        println!(
            "direct_v2_d20_setup={}",
            serde_json::to_string(&serde_json::json!({
                "setup_id": hex_digest(setup.setup_id()),
                "cycle_capacity": setup.config().cycle_capacity,
                "ram_k": setup.config().ram_k,
                "setup_total_micros": setup_total_micros,
                "metrics": setup_metrics,
                "memory": setup_memory,
            }))?
        );
        return Ok(());
    }

    eprintln!("direct_v2_d20_phase=prove_stream_start");
    let prove_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let prove_started = Instant::now();
    let (streaming, streaming_metrics) = prove_block_jolt_streaming_with_setup(
        &prepared.preprocessing,
        &setup,
        trace_blocks(&prepared, args.block_capacity),
    )?;
    let prove_stream_total_micros = prove_started.elapsed().as_micros();
    let prove_stream_memory = prove_memory_sampler.finish();
    eprintln!(
        "direct_v2_d20_phase=prove_stream_complete micros={} pcs_spool_bytes={}",
        prove_stream_total_micros, streaming_metrics.spooled_pcs_witness_bytes
    );

    eprintln!("direct_v2_d20_phase=finalization_start");
    let finalization_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let finalization_started = Instant::now();
    let (final_proof, finalization_metrics) =
        compress_block_jolt_final_proof_with_metrics(&setup, &streaming)?;
    let finalization_total_micros = finalization_started.elapsed().as_micros();
    let finalization_memory = finalization_memory_sampler.finish();
    let encoded = final_proof.to_bytes()?;
    eprintln!(
        "direct_v2_d20_phase=finalization_complete micros={} artifact_bytes={}",
        finalization_total_micros,
        encoded.len()
    );

    eprintln!("direct_v2_d20_phase=final_verify_start");
    let verify_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let verify_started = Instant::now();
    let decoded = jolt_core::zkvm::block::BlockJoltFinalProof::from_bytes(&encoded)?;
    decoded.verify(&setup)?;
    let final_decode_and_verify_micros = verify_started.elapsed().as_micros();
    let final_verify_memory = verify_memory_sampler.finish();
    eprintln!(
        "direct_v2_d20_phase=final_verify_complete micros={}",
        final_decode_and_verify_micros
    );
    let cycles_per_second = streaming_metrics.total_active_cycles as f64
        / (prove_stream_total_micros as f64 / 1_000_000.0);

    let mut artifact = D20BenchmarkArtifact {
        schema_version: D20_SCHEMA_VERSION,
        protocol_version: BLOCK_JOLT_PROTOCOL_VERSION.to_string(),
        platform: D20Platform {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            build_profile: args.build_profile_label,
            rustc: command_line("rustc", &["--version"]),
            git_commit: command_line("git", &["rev-parse", "HEAD"]),
            rayon_threads: std::env::var("RAYON_NUM_THREADS").ok(),
        },
        workload: prepared.workload,
        workload_sha3_256: hex_digest(prepared.workload_digest),
        block_capacity: args.block_capacity,
        derived_ram_k: setup.config().ram_k,
        setup_id: hex_digest(setup.setup_id()),
        preprocessing_micros: prepared.preprocessing_micros,
        sample: D20BenchmarkSample {
            verified: true,
            setup_total_micros,
            prove_stream_total_micros,
            finalization_total_micros,
            final_decode_and_verify_micros,
            cycles_per_second,
            setup_metrics,
            streaming_metrics,
            finalization_metrics,
            setup_memory,
            prove_stream_memory,
            finalization_memory,
            final_verify_memory,
        },
        comparison: None,
        artifact_digest: String::new(),
    };
    artifact.seal()?;
    if let Some(path) = &args.baseline_input {
        let baseline: D20BenchmarkArtifact = serde_json::from_slice(&std::fs::read(path)?)?;
        artifact.comparison = Some(
            artifact
                .compare(&baseline, args.max_regression_percent)
                .map_err(invalid_input)?,
        );
        artifact.seal()?;
    }
    artifact.validate().map_err(invalid_input)?;
    let markdown = args
        .markdown_output
        .clone()
        .unwrap_or_else(|| args.output.with_extension("md"));
    ensure_parent(&args.output)?;
    ensure_parent(&markdown)?;
    std::fs::write(
        &args.output,
        format!("{}\n", serde_json::to_string_pretty(&artifact)?),
    )?;
    std::fs::write(&markdown, artifact.markdown())?;
    println!("direct_v2_d20_json={}", args.output.display());
    println!("direct_v2_d20_markdown={}", markdown.display());
    println!("direct_v2_d20_digest={}", artifact.artifact_digest);
    if artifact
        .comparison
        .as_ref()
        .is_some_and(|item| !item.passed)
    {
        return Err(invalid_input("D21 regression threshold exceeded".to_string()).into());
    }
    Ok(())
}

fn prepare(args: &Args) -> Result<PreparedWorkload, Box<dyn Error>> {
    let started = Instant::now();
    let mut program = Program::new(args.workload.guest_name());
    args.workload.configure_program(&mut program);
    program.build(path_string(&args.guest_target)?);
    let inputs = args
        .workload
        .input_bytes(args.scale)
        .map_err(invalid_input)?;
    let (bytecode, init_memory_state, program_size, entry_address) = program.decode();
    let memory_config = args.workload.memory_config(program_size);
    let program_preprocessing = ProgramPreprocessing::<DoryCommitmentScheme>::preprocess(
        bytecode,
        init_memory_state,
        entry_address,
    )?;
    let shared = JoltSharedPreprocessing::new(
        program_preprocessing,
        MemoryLayout::new(&memory_config),
        args.workload.max_trace_length(),
    );
    let preprocessing = DirectChunkedPreprocessing::from_shared(&shared);
    let elf = program
        .get_elf_contents()
        .ok_or_else(|| invalid_input("guest ELF unavailable after build".to_string()))?;
    let workload = args.workload.id(args.scale);
    let mut hasher = Sha3_256::new();
    hasher.update(b"DIRECT_BLOCK_JOLT_D20_WORKLOAD_V1");
    hasher.update(workload.as_bytes());
    hasher.update(&elf);
    hasher.update(&inputs);
    let workload_digest = hasher.finalize().into();
    Ok(PreparedWorkload {
        program,
        elf,
        inputs,
        memory_config,
        preprocessing,
        workload,
        workload_digest,
        preprocessing_micros: started.elapsed().as_micros(),
    })
}

fn trace_blocks(
    prepared: &PreparedWorkload,
    block_capacity: usize,
) -> impl Iterator<Item = tracer::TraceBlock> + '_ {
    tracer::trace_blocks(
        &prepared.elf,
        prepared.program.elf.as_ref(),
        &prepared.inputs,
        &[],
        &[],
        &prepared.memory_config,
        None,
        block_capacity,
    )
}

fn validate_args(args: &Args) -> Result<(), String> {
    if [args.trace_audit_only, args.ram_audit_only, args.setup_only]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
        > 1
        || args.scale == 0
        || args.block_capacity < 2
        || !args.block_capacity.is_power_of_two()
        || args.build_profile_label.trim().is_empty()
        || args.memory_sample_interval_ms == 0
        || args.memory_sample_interval_ms > 1_000
        || !args.max_regression_percent.is_finite()
        || args.max_regression_percent < 0.0
    {
        return Err(
            "invalid mode, scale, block capacity, sampler interval, or regression limit".into(),
        );
    }
    Ok(())
}

fn digest_json(value: &D20BenchmarkArtifact) -> Result<[u8; 32], serde_json::Error> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha3_256::new();
    hasher.update(D20_DIGEST_DOMAIN);
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    Ok(hasher.finalize().into())
}

fn percent_change(current: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        if current == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (current - baseline) * 100.0 / baseline
    }
}

fn seconds(micros: u128) -> f64 {
    micros as f64 / 1_000_000.0
}

fn physical_memory_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

fn command_line(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn ensure_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn path_string(path: &Path) -> Result<&str, io::Error> {
    path.to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path is not UTF-8"))
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d20_percent_change_handles_improvement_and_zero() {
        assert_eq!(percent_change(80.0, 100.0), -20.0);
        assert_eq!(percent_change(0.0, 0.0), 0.0);
        assert!(percent_change(1.0, 0.0).is_infinite());
    }
}
