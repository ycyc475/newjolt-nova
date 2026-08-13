//! Minimal architecture audit: ELF -> streaming trace blocks -> direct seam.
//!
//! The production D7 entry additionally requires full direct preprocessing and
//! execution inputs; this diagnostic example only audits block construction.

use std::{env, error::Error, fs, path::PathBuf};

use common::jolt_device::MemoryConfig;
use jolt_core::zkvm::block::{
    DirectChunkedConfig, DirectChunkedPreprocessing, DirectChunkedProver,
};

fn main() -> Result<(), Box<dyn Error>> {
    let elf_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: direct_chunked_architecture <guest.elf> [block_capacity]")?;
    let block_capacity = env::args_os()
        .nth(2)
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(1 << 12);
    let elf = fs::read(&elf_path)?;
    let memory_config = MemoryConfig::default();
    let blocks = tracer::trace_blocks(
        &elf,
        Some(&elf_path),
        &[],
        &[],
        &[],
        &memory_config,
        None,
        block_capacity,
    );
    let direct = DirectChunkedProver::new(
        DirectChunkedPreprocessing::from_program_bytes(&elf, usize::MAX),
        DirectChunkedConfig {
            block_capacity,
            compress_final_spartan: false,
        },
    )?;

    let audit = direct.audit_trace_blocks(blocks)?;
    println!(
        "Direct architecture seam validated: streamed {} block(s), {} active cycles",
        audit.block_count, audit.total_cycles
    );
    Ok(())
}
