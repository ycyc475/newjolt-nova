//! D8 production benchmark.
//!
//! A validated Stage-19 artifact supplies native-Jolt and dual-data-flow
//! measurements for the exact same ELF/input digest. This runner executes only
//! the direct path: ELF -> bounded trace blocks -> block-native relations ->
//! Nova -> Dory/Spartan. It never constructs an `RV64IMACProver` or native
//! `JoltProof`.

use std::{
    error::Error,
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_core::{
    host::Program,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        block::{
            DirectChunkedConfig, DirectChunkedPreprocessing, DirectChunkedProver,
            DirectD8BenchmarkArtifact, DirectD8Measurement, DirectExecutionInputs,
            Stage19BenchmarkArtifact,
        },
        program::ProgramPreprocessing,
        verifier::JoltSharedPreprocessing,
    },
};
use sha3::{Digest, Sha3_256};

const DEFAULT_OUTPUT: &str = "benchmark-runs/direct-d8/fibonacci-2.json";

#[derive(Clone, Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 2)]
    scale: u32,

    #[arg(long, default_value_t = 2)]
    block_capacity: usize,

    /// Validated Stage-19 JSON produced on the same machine/build profile.
    #[arg(long)]
    stage19_baseline: PathBuf,

    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    #[arg(long)]
    markdown_output: Option<PathBuf>,

    #[arg(long, default_value = "target-direct-d8-guest")]
    guest_target: PathBuf,

    #[arg(long, default_value_t = 10)]
    memory_sample_interval_ms: u64,

    /// Validate the real lazy trace and hard-rechunk boundaries, then exit
    /// before relation proving. Useful before an expensive release benchmark.
    #[arg(long, default_value_t = false)]
    trace_audit_only: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("direct_stage8_error={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args).map_err(invalid_input)?;
    let baseline: Stage19BenchmarkArtifact =
        serde_json::from_slice(&std::fs::read(&args.stage19_baseline)?)?;
    baseline.validate().map_err(invalid_input)?;

    let workload = format!("fibonacci-{}", args.scale);
    let inputs = postcard::to_stdvec(&args.scale)?;
    let public_outputs = postcard::to_stdvec(&fibonacci(args.scale))?;
    let mut program = Program::new("fibonacci-guest");
    program.build(path_string(&args.guest_target)?);
    let (bytecode, init_memory_state, program_size, entry_address) = program.decode();
    let elf = program
        .get_elf_contents()
        .ok_or_else(|| invalid_input("guest ELF is unavailable after build".to_string()))?;
    let memory_config = MemoryConfig {
        program_size: Some(program_size),
        ..MemoryConfig::default()
    };
    let memory_layout = MemoryLayout::new(&memory_config);
    let program_preprocessing = ProgramPreprocessing::<DoryCommitmentScheme>::preprocess(
        bytecode,
        init_memory_state,
        entry_address,
    )?;
    let shared = JoltSharedPreprocessing::new(program_preprocessing, memory_layout, 1 << 16);
    let direct_preprocessing = DirectChunkedPreprocessing::from_shared(&shared);

    let workload_digest = workload_digest(&workload, &elf, &inputs);
    let workload_digest_hex = hex_digest(workload_digest);
    let baseline_sample = baseline
        .samples
        .iter()
        .find(|sample| {
            sample.workload == workload
                && sample.workload_sha3_256 == workload_digest_hex
                && sample.block_target_size == args.block_capacity
        })
        .ok_or_else(|| {
            invalid_input(format!(
                "Stage-19 baseline has no matching {workload}, digest {workload_digest_hex}, block capacity {} sample",
                args.block_capacity
            ))
        })?;

    let prover = DirectChunkedProver::new(
        direct_preprocessing,
        DirectChunkedConfig {
            block_capacity: args.block_capacity,
            compress_final_spartan: true,
        },
    )?;
    if args.trace_audit_only {
        let blocks = tracer::trace_blocks(
            &elf,
            program.elf.as_ref(),
            &inputs,
            &[],
            &[],
            &memory_config,
            None,
            args.block_capacity,
        );
        let (audit, metrics) = prover.audit_production_trace(blocks)?;
        println!("direct_d8_trace_audit_blocks={}", audit.block_count);
        println!("direct_d8_trace_audit_cycles={}", audit.total_cycles);
        println!(
            "direct_d8_trace_audit_digest={}",
            hex_digest(audit.trace_digest)
        );
        println!("direct_d8_trace_metrics={}", metrics.to_json_pretty()?);
        return Ok(());
    }
    let sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let prove_start = Instant::now();
    let blocks = tracer::trace_blocks(
        &elf,
        program.elf.as_ref(),
        &inputs,
        &[],
        &[],
        &memory_config,
        None,
        args.block_capacity,
    );
    let (proof, metrics) = prover.prove_with_metrics(
        DirectExecutionInputs {
            public_inputs: inputs,
            public_outputs,
            trusted_advice: Vec::new(),
            untrusted_advice: Vec::new(),
            panic: false,
        },
        blocks,
    )?;
    let prove_micros = elapsed_micros(prove_start);
    let verify_start = Instant::now();
    prover.verify(&proof)?;
    let verify_micros = elapsed_micros(verify_start);
    let memory = sampler.finish();
    let proof_size = proof.benchmark_size_breakdown();
    let direct = DirectD8Measurement {
        proof_verified: true,
        proof_size,
        prove_micros,
        verify_micros,
        peak_rss_delta_bytes: memory.peak_delta,
        metrics,
    };
    let artifact = DirectD8BenchmarkArtifact::from_stage19_sample(baseline_sample, direct)
        .map_err(invalid_input)?;

    ensure_parent(&args.output)?;
    let markdown = args
        .markdown_output
        .clone()
        .unwrap_or_else(|| args.output.with_extension("md"));
    ensure_parent(&markdown)?;
    std::fs::write(
        &args.output,
        artifact.to_json_pretty().map_err(invalid_input)?,
    )?;
    std::fs::write(&markdown, artifact.to_markdown().map_err(invalid_input)?)?;
    println!("direct_d8_json={}", args.output.display());
    println!("direct_d8_markdown={}", markdown.display());
    println!("direct_d8_digest={}", artifact.artifact_digest);
    Ok(())
}

fn fibonacci(n: u32) -> u128 {
    let mut a = 0u128;
    let mut b = 1u128;
    for _ in 1..n {
        let sum = a.wrapping_add(b);
        a = b;
        b = sum;
    }
    b
}

fn workload_digest(workload: &str, elf: &[u8], inputs: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_STAGE19_WORKLOAD_V1");
    hasher.update(workload.as_bytes());
    hasher.update(elf);
    hasher.update(inputs);
    hasher.finalize().into()
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

fn validate_args(args: &Args) -> Result<(), String> {
    if args.scale == 0 {
        return Err("scale must be non-zero".to_string());
    }
    if args.block_capacity < 2 || !args.block_capacity.is_power_of_two() {
        return Err("block-capacity must be a power of two of at least two".to_string());
    }
    if args.memory_sample_interval_ms == 0 || args.memory_sample_interval_ms > 1_000 {
        return Err("memory-sample-interval-ms must be in 1..=1000".to_string());
    }
    if args.output == args.stage19_baseline {
        return Err("output must not overwrite the Stage-19 baseline".to_string());
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct MemoryMeasurement {
    peak_delta: Option<u64>,
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

    fn finish(mut self) -> MemoryMeasurement {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let peak = self
            .start
            .map(|start| self.peak.load(Ordering::Relaxed).max(start));
        MemoryMeasurement {
            peak_delta: peak.zip(self.start).map(|(peak, start)| peak - start),
        }
    }
}

fn physical_memory_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().max(1).min(u64::MAX as u128) as u64
}

fn ensure_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn path_string(path: &Path) -> Result<&str, io::Error> {
    path.to_str()
        .ok_or_else(|| invalid_input(format!("path is not valid UTF-8: {}", path.display())))
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d8_fibonacci_host_output_matches_guest_definition() {
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn d8_benchmark_rejects_invalid_capacity() {
        let args = Args {
            scale: 2,
            block_capacity: 3,
            stage19_baseline: PathBuf::from("baseline.json"),
            output: PathBuf::from("direct.json"),
            markdown_output: None,
            guest_target: PathBuf::from("target"),
            memory_sample_interval_ms: 10,
            trace_audit_only: false,
        };
        assert!(validate_args(&args).is_err());
    }
}
